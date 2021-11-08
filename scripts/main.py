import os
import sys
import yaml
import time
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets, transforms

import helper
import augmentations
import build_models
import build_datasets

import warnings
warnings.filterwarnings("ignore")

def parse_args(params_path=None):
    parser = argparse.ArgumentParser(description='DINO reimplementation')

    if params_path == None:
        parser.add_argument('--params_path', type=str, required=False, help='Give a valid yaml file that contains all params to load.')
        params_path = parser.parse_args().params_path

    with open(params_path) as f:
        args = yaml.safe_load(f)
    
    print(end='\n\n'+'=='*50+'\n\n')
    print('ARGS ARE: ')
    print(args, end='\n\n'+'=='*50+'\n\n')
    return args

def train_process(rank, args, start_training=True):
    print(f'Rank: {rank}. Start preparing for training ')

    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    augmentation_params = args['augmentation_params']

    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']

    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params)

    # ============ preparing data ... ============
    # Set up data loader with augmentations
    # Load an example Dataset with augmentations
    transforms_aug = augmentations.DataAugmentationDINO(
        augmentation_params['global_crops_scale'],
        augmentation_params['local_crops_scale'],
        augmentation_params['local_crops_number'],
        augmentation_params['global_size'],
        augmentation_params['local_size']
    )
    transforms_plain = transforms_aug.transforms_plain
    
    start = time.time()
    # The aug_dataset is for training, while plain_datasets is mostly used to compute knn and visualize the embeddings
    train_aug_dataset = build_datasets.get_datasets(dataset_params, 'train/', transforms_aug)
    train_plain_dataset = build_datasets.get_datasets(dataset_params, 'train/', transforms_plain)
    val_plain_dataset = build_datasets.get_datasets(dataset_params, 'val/', transforms_plain)

    if train_plain_dataset.classes != val_plain_dataset.classes:
        raise ValueError("Inconsistent classes in train and val.")
    
    if rank == system_params['num_gpus']-1:
        print(f"On each rank, data loaded: there are {len(train_aug_dataset)} train images. ")
        print(f"On each rank, data loaded: there are {len(train_plain_dataset)} train images. ")
        print(f"On each rank, data loaded: there are {len(val_plain_dataset)} val images. ")

        print(f'Building the train_aug_dataset, train_plain_dataset and val_plain_dataset took {time.time() - start} seconds')
        print()

    # Set sampler that restricts data loading to a subset of the dataset
    # In conjunction with torch.nn.parallel.DistributedDataParallel
    train_aug_sampler = torch.utils.data.DistributedSampler(train_aug_dataset, shuffle=True)
    train_plain_sampler = torch.utils.data.DistributedSampler(train_plain_dataset, shuffle=True)
    val_plain_sampler = torch.utils.data.DistributedSampler(val_plain_dataset, shuffle=True)

    # Prepare the data for training with DataLoaders
    # pin_memory makes transferring images from CPU to GPU faster
    train_aug_dataloader = DataLoader(train_aug_dataset, sampler=train_aug_sampler, batch_size=int(trainloader_params['batch_size']/system_params['num_gpus']),
                                  num_workers=trainloader_params['num_workers'], pin_memory=trainloader_params['pin_memory'], drop_last=trainloader_params['drop_last'])
    train_plain_dataloader = DataLoader(train_plain_dataset, sampler=train_plain_sampler, batch_size=int(trainloader_params['batch_size']/system_params['num_gpus']),
                                      num_workers=trainloader_params['num_workers'], pin_memory=trainloader_params['pin_memory'], drop_last=trainloader_params['drop_last'])
    val_plain_dataloader = DataLoader(val_plain_dataset, sampler=val_plain_sampler, batch_size=int(valloader_params['batch_size']/system_params['num_gpus']),
                                        num_workers=valloader_params['num_workers'], pin_memory=valloader_params['pin_memory'], drop_last=valloader_params['drop_last'])

    if rank == system_params['num_gpus']-1:
        print(f"On each rank, data loaded: there are {len(train_aug_dataloader)} train_dataloaders. ")
        print(f"On each rank, data loaded: there are {len(train_plain_dataloader)} train_plain_dataloader. ")
        print(f"On each rank, data loaded: there are {len(val_plain_dataloader)} val_plain_dataloader. ", end='\n\n')

    # ============ building student and teacher networks ... ============
    print(f"Rank: {rank}. Creating model: {model_params['backbone_option']}", end='\n\n')
    student_backbone, student_head, teacher_backbone, teacher_head = build_models.build_dino(model_params)

    student = build_models.MultiCropWrapper(student_backbone, student_head)
    teacher = build_models.MultiCropWrapper(teacher_backbone, teacher_head)

    # Move networks to gpu
    # This step is necessary for DDP later
    student, teacher = student.cuda(),teacher.cuda()

    # Synchronize batch norms (if any)
    if helper.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # We need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[rank])
        teacher_without_ddp = teacher.module
    else:
        # Teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[rank])

    # Teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # There is no backpropagation through the teacher, so no need for gradients
    # This step is to save some memory
    for param in teacher.parameters():
        param.requires_grad = False

    # ============ preparing loss ... ============



    # model = teacher
    # x = torch.randn(1, 3, 28, 28)
    # x = x.repeat(2, 1, 1, 1).cuda(non_blocking=True)
    # # print(x)
    # print(x.shape)
    # y = model(x)
    # print(y)
    # print(y[0].shape)
    #
    # model = student
    # x = torch.randn(1, 3, 28, 28)
    # x = x.repeat(2, 1, 1, 1).cuda(non_blocking=True)
    # # print(x)
    # print(x.shape)
    # y = model(x)
    # print(y)
    # print(y[0].shape)

    return

def main(rank, args):
    # Set up training
    train_process(rank, args, start_training=True)

if __name__ == '__main__':
    # Read params and print them
    args = parse_args(params_path='yaml/test_params.yaml')
    # print(int(os.environ["RANK"]))
    # print(int(os.environ['WORLD_SIZE']))
    # print(int(os.environ['LOCAL_RANK']))

    # Launch multi-gpu / distributed training
    helper.launch(main, args)