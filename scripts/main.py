import os
import sys
import time
import datetime
import yaml
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import helper
import evaluation
import prepare_models
import prepare_datasets
import prepare_losses
import prepare_trainers

import warnings

warnings.filterwarnings("ignore")

def parse_args(params_path=None):
    parser = argparse.ArgumentParser(description='DINO reimplementation')

    if params_path == None:
        parser.add_argument('--params_path', type=str, required=False,
                            help='Give a valid yaml file that contains all params to load.')
        params_path = parser.parse_args().params_path

    with open(params_path) as f:
        args = yaml.safe_load(f)

    print(end='\n\n' + '==' * 50 + '\n\n')
    print('ARGS ARE: ')
    print(json.dumps(args, indent=4))
    return args

def knn_in_train_process(rank, writer, use_cuda, backbone, train_dataloader, val_dataloader, label_mapping, epoch, save_params, if_original=False, if_eval=False):
    knn_start_time = time.time()
    train_embeddings, train_labels = evaluation.compute_embedding(backbone.module.backbone, train_dataloader,
                                                                     label_mapping, use_cuda=use_cuda, return_tb=False)
    val_embeddings, val_labels = evaluation.compute_embedding(backbone.module.backbone, val_dataloader,
                                                                 label_mapping, use_cuda=use_cuda, return_tb=False)
    if rank==0:
        train_embeddings = nn.functional.normalize(train_embeddings, dim=1, p=2).cuda()
        val_embeddings = nn.functional.normalize(val_embeddings, dim=1, p=2).cuda()
        print(f'train/val embeddings size: , {train_embeddings.size()}, {val_embeddings.size()}')
        train_labels = train_labels.long().cuda()
        val_labels = val_labels.long().cuda()
        print(f'train/val labels size: , {train_labels.size()}, {val_labels.size()}')
        print("Features are ready!\nStart the k-NN classification.")

        for k in save_params['nb_knn']:
            top1, top5 = evaluation.knn_classifier(train_embeddings, train_labels,
                                                   val_embeddings, val_labels, k, save_params['temp_knn'])
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
            knn_total_time = time.time() - knn_start_time
            knn_total_time_str = str(datetime.timedelta(seconds=int(knn_total_time)))
            print('KNN time {}'.format(knn_total_time_str))
            if if_original:
                writer.add_scalar(f"{k}nn_top1_original", top1, epoch)
                writer.add_scalar(f"{k}nn_top5_original", top5, epoch)
            else:
                if if_eval:
                    writer.add_scalar(f"{k}nn_top1_eval", top1, epoch)
                    writer.add_scalar(f"{k}nn_top5_eval", top5, epoch)
                else:
                    writer.add_scalar(f"{k}nn_top1", top1, epoch)
                    writer.add_scalar(f"{k}nn_top5", top5, epoch)

    return train_embeddings, train_labels, val_embeddings, val_labels

def train_process(rank, args, start_training=True):
    # Define some parameters for easy access
    save_params = args['save_params']
    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    augmentation_params = args['augmentation_params']
    training_params = args['training_params']
    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']

    # Tensorboard Summarywriter for logging
    # Log network input arg in Tensorboard
    tensorboard_path = os.path.join(save_params['output_dir'], 'tensorboard/')
    if rank==0:
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
        global  writer
    writer = SummaryWriter(tensorboard_path)
    writer.add_text("Configuration", json.dumps(args, indent=4))

    # ============ Preparing system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params)

    # ============ Preparing data ... ============
    # Define transformations applied on augmented and plain data
    # The aug_dataset is for training, while plain_datasets is mostly used to compute knn and visualize the embeddings
    transforms_aug = prepare_datasets.DataAugmentationDINO(
        augmentation_params['global_crops_scale'], augmentation_params['local_crops_scale'],
        augmentation_params['local_crops_number'], augmentation_params['global_size'],
        augmentation_params['local_size'])
    transforms_plain = transforms_aug.transforms_plain

    start_time = time.time()
    label_mapping = prepare_datasets.GetDatasets(dataset_params).label_mapping
    train_aug_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('train/', transforms_aug)
    train_plain_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('train/', transforms_plain, include_index=True)
    val_plain_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('val/', transforms_plain, include_index=True)

    if train_plain_dataset.classes != val_plain_dataset.classes:
        raise ValueError("Inconsistent classes in train and val.")
    if rank==0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f'Building the train_aug_dataset, train_plain_dataset and val_plain_dataset took {total_time_str} seconds',
            end='\n\n')

    # Set sampler that restricts data loading to a subset of the dataset
    # In conjunction with torch.nn.parallel.DistributedDataParallel
    train_aug_sampler = torch.utils.data.DistributedSampler(train_aug_dataset, shuffle=True)
    train_plain_sampler = torch.utils.data.DistributedSampler(train_plain_dataset, shuffle=True)
    val_plain_sampler = torch.utils.data.DistributedSampler(val_plain_dataset, shuffle=True)

    # Prepare the data for training with DataLoaders
    # pin_memory makes transferring images from CPU to GPU faster
    train_aug_dataloader = DataLoader(train_aug_dataset, sampler=train_aug_sampler,
                                      batch_size=int(trainloader_params['batch_size'] / system_params['num_gpus']),
                                      num_workers=trainloader_params['num_workers'],
                                      pin_memory=trainloader_params['pin_memory'],
                                      drop_last=trainloader_params['drop_last'])
    train_plain_dataloader = DataLoader(train_plain_dataset, sampler=train_plain_sampler,
                                        batch_size=int(trainloader_params['batch_size'] / system_params['num_gpus']),
                                        num_workers=trainloader_params['num_workers'],
                                        pin_memory=trainloader_params['pin_memory'],
                                        drop_last=trainloader_params['drop_last'])
    val_plain_dataloader = DataLoader(val_plain_dataset, sampler=val_plain_sampler,
                                      batch_size=int(valloader_params['batch_size'] / system_params['num_gpus']),
                                      num_workers=valloader_params['num_workers'],
                                      pin_memory=valloader_params['pin_memory'],
                                      drop_last=valloader_params['drop_last'])
    if rank==0:
        print(f"There are {len(train_aug_dataloader)} train_dataloaders on each rank. ")
        print(f"There are {len(train_plain_dataloader)} train_plain_dataloader on each rank. ")
        print(f"There are {len(val_plain_dataloader)} val_plain_dataloader on each rank. ")

    # ============ Building student and teacher networks ... ============
    if rank==0:
        print(f"Rank: {rank}. Creating model: {model_params['backbone_option']}", end='\n\n')
    student_backbone, student_head, teacher_backbone, teacher_head = prepare_models.build_dino(model_params)
    student = prepare_models.MultiCropWrapper(student_backbone, student_head)
    teacher = prepare_models.MultiCropWrapper(teacher_backbone, teacher_head)

    # Move networks to gpu. This step is necessary for DDP later
    student, teacher = student.cuda(), teacher.cuda()

    # If there is any batch norms, we synchronize them
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

    # Teacher and student start with the same weights and we turn off the gradients on teacher network
    helper.initialize_momentum_state(online_net=student, momentum_net=teacher_without_ddp)

    # Log the number of trainable parameters in Tensorboard
    if rank==0:
        n_parameters_student = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print('Number of params of the student:', n_parameters_student)
        writer.add_text("Number of params of the student", str(n_parameters_student))

    # ============ Preparing loss ... ============
    # Move dino_loss to gpu
    dino_loss = prepare_losses.DINOLoss(out_dim=model_params['out_dim'],
        num_crops=augmentation_params['local_crops_number'] + 2,
        warmup_teacher_temp=training_params['warmup_teacher_temp'], teacher_temp=training_params['teacher_temp'],
        warmup_teacher_temp_epochs=training_params['warmup_teacher_temp_epochs'],
        num_epochs=training_params['num_epochs'], student_temp=training_params['student_temp'],
        center_momentum=training_params['center_momentum']).cuda()

    # ============ Preparing optimizer ... ============
    optimizer = prepare_trainers.get_optimizer(optimizer_choice=training_params['optimizer']['name'],
        params_dict=helper.get_params_groups(student), lr=training_params['optimizer']['sgd']['lr'],
        momentum=training_params['optimizer']['sgd']['momentum'])
    # For mixed precision training
    # Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters,
    # so the scale factor does not interfere with the learning rate.
    fp16_scaler = None
    if system_params['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ Initialize schedulers ... ============
    # Linear scaling rule according to the paper below
    # "Accurate, large minibatch sgd: Training imagenet in 1 hour."
    base_lr = training_params['lr']['base_lr'] * trainloader_params['batch_size'] / 256.
    lr_schedule = helper.cosine_scheduler(base_value=base_lr, final_value=training_params['lr']['final_lr'],
        epochs=training_params['num_epochs'], niter_per_ep=len(train_aug_dataset),
        warmup_epochs=training_params['lr']['warmup_epochs'],
        start_warmup_value=training_params['lr']['start_warmup_lr'], )
    wd_schedule = helper.cosine_scheduler(base_value=training_params['wd']['base_wd'],
        final_value=training_params['wd']['final_wd'], epochs=training_params['num_epochs'],
        niter_per_ep=len(train_aug_dataset), )
    # Momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = helper.cosine_scheduler(base_value=training_params['momentum']['base_momentum_teacher'],
        final_value=training_params['momentum']['final_momentum_teacher'], epochs=training_params['num_epochs'],
        niter_per_ep=len(train_aug_dataset))
    if rank==0:
        print(f"Loss, optimizer and schedulers ready.")

    # ============ Optionally resume training ... ============
    to_restore = {'epoch': save_params['restore_epoch']}
    helper.restart_from_checkpoint(os.path.join(save_params['output_dir'], "checkpoint.pth"), run_variables=to_restore,
        student=student, teacher=teacher, optimizer=optimizer, fp16_scaler=fp16_scaler, dino_loss=dino_loss, )

    if start_training:
        print("Starting DINO training !")
        # Record the starting time
        start_time = time.time()

        if dataset_params['dataset_name'] == 'ImageNet':
            use_cuda = False
        else:
            use_cuda = False

        if save_params['tb_logoriginal']:
            # ============ Adding embeddings in tensorboard before the training starts ... ============
            tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(student.module.backbone,
                                                                                     val_plain_dataloader, label_mapping, use_cuda=use_cuda,
                                                                                     return_tb=True, subset_size=100)
            if rank==0:
                writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img, global_step=to_restore['epoch'],
                                     tag="embeddings_original")

            # ============ Adding knn results in tensorboard before the training starts ... ============
            _, _, _, _ = knn_in_train_process(rank=rank, writer=writer, use_cuda=use_cuda, backbone=student, train_dataloader=train_plain_dataloader, val_dataloader=val_plain_dataloader,
                                 label_mapping=label_mapping, epoch=to_restore['epoch'], save_params=save_params, if_original=True)


        for epoch in range(to_restore['epoch'], training_params['num_epochs']):
            # In distributed mode, calling the :meth:`set_epoch` method at
            # the beginning of each epoch **before** creating the :class:`DataLoader` iterator
            # is necessary to make shuffling work properly across multiple epochs. Otherwise,
            # the same ordering will be always used.
            if isinstance(train_aug_dataloader.sampler, torch.utils.data.DistributedSampler):
                train_aug_dataloader.sampler.set_epoch(epoch)
                train_plain_dataloader.sampler.set_epoch(epoch)
                val_plain_dataloader.sampler.set_epoch(epoch)


            # ============ Training one epoch of DINO ... ============
            train_stats = prepare_trainers.kd_train_one_epoch(epoch, training_params['num_epochs'], student, teacher,
                                                              teacher_without_ddp, dino_loss, train_aug_dataloader,
                                                              optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                                              training_params['clip_grad'],
                                                              training_params['freeze_last_layer'], fp16_scaler)

            # ============ Writing logs & adding representation embeddings & KNN results in tensorboard ... ============
            # Log the number of training loss in Tensorboard, at every epoch
            if rank==0:
                writer.add_scalar("train_loss", train_stats['loss'], epoch)
            # Log the embeddings & KNN results in Tensorboard, at every tb_freq epoch
            if save_params['tb_freq'] and epoch % save_params['tb_freq'] == 0:
                tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(student.module.backbone,
                                                                                         val_plain_dataloader,
                                                                                         label_mapping,
                                                                                         return_tb=True,
                                                                                         subset_size=100)
                if rank==0:
                    writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img,
                                         global_step=epoch, tag="embeddings")



                train_embeddings, train_labels, val_embeddings, val_labels = knn_in_train_process(rank=rank, writer=writer, use_cuda=use_cuda, backbone=student, train_dataloader=train_plain_dataloader, val_dataloader=val_plain_dataloader,
                                     label_mapping=label_mapping, epoch=epoch, save_params=save_params)
                # If the current epoch is the last epoch, we save the embeddings 
                if rank==0 and epoch == training_params['num_epochs']-1:
                    torch.save(train_embeddings.cpu(), os.path.join(save_params['output_dir'], f"trainembeddings{epoch:04}.pth"))
                    torch.save(val_embeddings.cpu(), os.path.join(save_params['output_dir'], f"valembeddings{epoch:04}.pth"))
                    torch.save(train_labels.cpu(), os.path.join(save_params['output_dir'], f"trainlabels{epoch:04}.pth"))
                    torch.save(val_labels.cpu(), os.path.join(save_params['output_dir'], f"vallabels{epoch:04}.pth"))


            # ============ Saving models and writing logs in log.txt file ... ============
            save_dict = {'student': student.state_dict(), 'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args,
                'dino_loss': dino_loss.state_dict(), }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            if rank==0:
                torch.save(save_dict, os.path.join(save_params['output_dir'], "checkpoint.pth"))
                if save_params['saveckp_freq'] and epoch % save_params['saveckp_freq'] == 0:
                    torch.save(save_dict, os.path.join(save_params['output_dir'], f'checkpoint{epoch:04}.pth'))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            if rank==0:
                with (Path(save_params['output_dir']) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # Log the number of total training time in Tensorboard
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if rank==0:
            with (Path(save_params['output_dir']) / "log.txt").open("a") as f:
                f.write(f"Training time from epoch {to_restore['epoch']} to epoch {training_params['num_epochs']}: " + total_time_str + "\n")
    else:
        epoch = to_restore['epoch']-1
        print("Starting DINO evaluation at epoch ", str(epoch))
        # Record the starting time
        start_time = time.time()
        checkpoint = torch.load(os.path.join(save_params['output_dir'], f"checkpoint{epoch:04}.pth"))
        student.load_state_dict(checkpoint['student'])
        tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(student.module.backbone,
                                                                                 val_plain_dataloader,
                                                                                 label_mapping,
                                                                                 return_tb=True,
                                                                                 subset_size=100)
        if rank==0:
            writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img,
                                 global_step=epoch, tag="embeddings_eval")

        _, _, _, _ = knn_in_train_process(rank=rank, writer=writer, use_cuda=use_cuda, backbone=student, train_dataloader=train_plain_dataloader, val_dataloader=val_plain_dataloader,
                             label_mapping=label_mapping, epoch=epoch, save_params=save_params, if_eval=True)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaludation time {}'.format(total_time_str))
        
def main(rank, args):
    # Set up training
    train_process(rank, args, start_training=args['start_training'])


if __name__ == '__main__':
    # Read params and print them
    args = parse_args(params_path='yaml/test_params.yaml')

    # Launch multi-gpu / distributed training
    helper.launch(main, args)


