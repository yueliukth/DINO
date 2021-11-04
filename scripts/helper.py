import os
import sys
import math
import numpy as np
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

def get_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def set_sys_params(params):
    # Set gpu params
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_ids']
    # Set random seeds for reproducibility
    seed = params['random_seed'] + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_rank():
    if not ddp():
        return 0
    return dist.get_rank()

def ddp():
    world_size = dist.get_world_size()
    if not dist.is_available() or not dist.is_initialized() or world_size < 2:
        return False
    return True

def run_distributed_workers(rank, main_func, world_size, dist_url, args):
    # Initialize the process group
    dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank)

    # Synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    if ddp():
        dist.barrier()

    torch.cuda.set_device(rank)
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True, end='\n\n'+'=='*50+'\n\n')

    main_func(rank, args)

def launch(main_func, args=()):
    system_params = args['system_params']
    world_size = system_params['num_gpus']
    port = get_open_port()
    dist_url = f"tcp://127.0.0.1:{port}"

    mp.spawn(
        run_distributed_workers,
        nprocs=world_size,
        args=(main_func, world_size, dist_url, args),
        daemon=False,
    )

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def print_layers(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    print(names)

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)