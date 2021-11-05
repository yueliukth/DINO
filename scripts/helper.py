import os
import sys
import math
import numpy as np
import socket
import torch
import torch.backends.cudnn as cudnn
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
    # Set random seeds for reproducibility TODO: to figure out whether it is necessary to have different random seeds
    # on different ranks (DeiT uses different seeds) 
    seed = params['random_seed'] #+ get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

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
        rank, dist_url), flush=True)

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


