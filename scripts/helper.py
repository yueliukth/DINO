import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def ddp_is_on():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    world_size = dist.get_world_size()
    if world_size < 2:
        return False
    return True

def synchronize():
    if not ddp_is_on():
        return
    dist.barrier()

def set_sys_params(params):
    # Set gpu params
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_ids']
    # Set random seeds
    torch.manual_seed(params['random_seed'])
    torch.cuda.manual_seed_all(params['random_seed'])
    np.random.seed(params['random_seed'])

def _distributed_worker(rank, main_func, world_size, dist_url, args):
    dist.init_process_group(backend="NCCL", init_method=dist_url, world_size=world_size, rank=rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    synchronize()

    torch.cuda.set_device(rank)
    main_func(args)

def launch(main_func, args=()):
    system_params = args['system_params']
    set_sys_params(system_params)

    world_size = system_params['num_gpus']
    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"

    mp.spawn(
        _distributed_worker,
        nprocs=world_size,
        args=(main_func, world_size, dist_url, args),
        daemon=False,
    )


def print_seperate_line():
    print()
    print()
    for i in range(50):
        sys.stdout.write("==")
    print()
    print()