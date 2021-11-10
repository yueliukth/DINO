import os
import sys
import time
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
import prepare_augmentations
import prepare_models
import prepare_datasets
import prepare_losses

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

    ckpt_params = args['ckpt_params']
    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    augmentation_params = args['augmentation_params']
    training_params = args['training_params']

    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']

    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params)

    # ============ preparing data ... ============
    # Set up data loader with augmentations
    # Load an example Dataset with augmentations
    transforms_aug = prepare_augmentations.DataAugmentationDINO(
        augmentation_params['global_crops_scale'],
        augmentation_params['local_crops_scale'],
        augmentation_params['local_crops_number'],
        augmentation_params['global_size'],
        augmentation_params['local_size']
    )
    transforms_plain = transforms_aug.transforms_plain
    
    start = time.time()
    # The aug_dataset is for training, while plain_datasets is mostly used to compute knn and visualize the embeddings
    train_aug_dataset = prepare_datasets.get_datasets(dataset_params, 'train/', transforms_aug)
    train_plain_dataset = prepare_datasets.get_datasets(dataset_params, 'train/', transforms_plain)
    val_plain_dataset = prepare_datasets.get_datasets(dataset_params, 'val/', transforms_plain)

    if train_plain_dataset.classes != val_plain_dataset.classes:
        raise ValueError("Inconsistent classes in train and val.")

    if rank == system_params['num_gpus']-1:
        print(f"On each rank: there are {len(train_aug_dataset)} train images. ")
        print(f"On each rank: there are {len(train_plain_dataset)} train images. ")
        print(f"On each rank: there are {len(val_plain_dataset)} val images. ")

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
        print(f"On each rank: there are {len(train_aug_dataloader)} train_dataloaders. ")
        print(f"On each rank: there are {len(train_plain_dataloader)} train_plain_dataloader. ")
        print(f"On each rank: there are {len(val_plain_dataloader)} val_plain_dataloader. ", end='\n\n')

    # ============ building student and teacher networks ... ============
    print(f"Rank: {rank}. Creating model: {model_params['backbone_option']}", end='\n\n')
    student_backbone, student_head, teacher_backbone, teacher_head = prepare_models.build_dino(model_params)

    student = prepare_models.MultiCropWrapper(student_backbone, student_head)
    teacher = prepare_models.MultiCropWrapper(teacher_backbone, teacher_head)

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
    # Move dino_loss to gpu
    dino_loss = prepare_losses.DINOLoss(
        out_dim=model_params['out_dim'],
        num_crops=augmentation_params['local_crops_number']+2,
        warmup_teacher_temp=training_params['warmup_teacher_temp'],
        teacher_temp=training_params['teacher_temp'],
        warmup_teacher_temp_epochs=training_params['warmup_teacher_temp_epochs'],
        num_epochs=training_params['num_epochs'],
        student_temp=training_params['student_temp'],
        center_momentum=training_params['center_momentum']).cuda()

    # ============ preparing optimizer ... ============
    params_dict = helper.get_params_groups(student)
    if training_params['optimizer']['name'] == "adamw":
        optimizer = torch.optim.AdamW(params_dict)  # to use with ViTs
    elif training_params['optimizer']['name'] == "sgd":
        optimizer = torch.optim.SGD(params_dict, lr=training_params['optimizer']['sgd']['lr'], momentum=training_params['optimizer']['sgd']['momentum'])  # lr is set by scheduler
    elif training_params['optimizer']['name'] == "lars":
        optimizer = utils.LARS(params_dict)  # to use with convnet and large batches
    # For mixed precision training
    # Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters,
    # so the scale factor does not interfere with the learning rate.
    fp16_scaler = None
    if system_params['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = helper.cosine_scheduler(
        base_value=training_params['lr']['base_lr'] * trainloader_params['batch_size'] / 256.,  # linear scaling rule according to the paper below
                                                                                          # "Accurate, large minibatch sgd: Training imagenet in 1 hour."
        final_value=training_params['lr']['final_lr'],
        epochs=training_params['num_epochs'],
        niter_per_ep=len(train_aug_dataset),
        warmup_epochs=training_params['lr']['warmup_epochs'],
        start_warmup_value=training_params['lr']['start_warmup_lr'],
        )
    wd_schedule = helper.cosine_scheduler(
        base_value=training_params['wd']['base_wd'],
        final_value=training_params['wd']['final_wd'],
        epochs=training_params['num_epochs'],
        niter_per_ep=len(train_aug_dataset),
    )
    # Momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = helper.cosine_scheduler(base_value=training_params['momentum']['base_momentum_teacher'],
                                                final_value=training_params['momentum']['final_momentum_teacher'],
                                                epochs=training_params['num_epochs'],
                                                niter_per_ep=len(train_aug_dataset))
    if rank == system_params['num_gpus']-1:
        print(f"Loss, optimizer and schedulers ready.")


    # ============ start the training process ... ============
    # ============ optionally resume training ... ============
    to_restore = {'epoch': ckpt_params['restore_epoch']}
    helper.restart_from_checkpoint(
        os.path.join(ckpt_params['output_dir'], "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(to_restore['epoch'], training_params['num_epochs']):
        # In distributed mode, calling the :meth:`set_epoch` method at
        # the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        # is necessary to make shuffling work properly across multiple epochs. Otherwise,
        # the same ordering will be always used.
        train_aug_dataloader.sampler.set_epoch(epoch)
        train_plain_dataloader.sampler.set_epoch(epoch)
        val_plain_dataloader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        metric_logger = helper.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, training_params['num_epochs'])
        for it, (images, _) in enumerate(metric_logger.log_every(iterable=train_aug_dataloader, print_freq=10, header=header)):

            images = [im.cuda(non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                student_output = student(images)
                print(len(teacher_output))
                # Each rank has size of teacher and student outputs: torch.Size([BATCH_SIZE/NUM_GPUS*NUM_GLOBAL_CROPS, OUT_DIM]), torch.Size([BATCH_SIZE/NUM_GPUS*NUM_ALL_CROPS, OUT_DIM])
         
         
            
        optimizer.zero_grad() 
        train_loss = dino_loss.forward(student_output, teacher_output, epoch)
        train_loss.backward() #retain_graph=True
        # student update
        optimizer.step()
        # teacher update
        teacher.parameters() = momentum_schedule[epoch]*teacher.parameters() + (1-momentum_schedule[epoch])*student.parameters() 
        
        



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

    # Launch multi-gpu / distributed training
    helper.launch(main, args)
