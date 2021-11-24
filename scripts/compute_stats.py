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

@torch.no_grad()
def knn_with_features(writer, train_embeddings, train_labels, val_embeddings, val_labels, epoch, save_params,
                      if_original=False, if_eval=False):
    for k in save_params['nb_knn']:
        top1, top5 = evaluation.knn_classifier(train_embeddings, train_labels, val_embeddings, val_labels, k,
                                               save_params['temp_knn'])
        print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        if writer != None:
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

@torch.no_grad()
def knn_in_train_process(rank, writer, use_cuda, backbone, train_dataloader, val_dataloader, label_mapping, epoch,
                         save_params, if_original=False, if_eval=False):
    knn_start_time = time.time()
    train_embeddings, train_labels = evaluation.compute_embedding(backbone, train_dataloader, label_mapping,
                                                                  use_cuda=use_cuda, return_tb=False)
    val_embeddings, val_labels = evaluation.compute_embedding(backbone, val_dataloader, label_mapping,
                                                              use_cuda=use_cuda, return_tb=False)
    if rank == 0:
        train_embeddings = nn.functional.normalize(train_embeddings, dim=1, p=2).cuda()
        val_embeddings = nn.functional.normalize(val_embeddings, dim=1, p=2).cuda()
        print(f'train/val embeddings size: , {train_embeddings.size()}, {val_embeddings.size()}')
        train_labels = train_labels.long().cuda()
        val_labels = val_labels.long().cuda()
        print(f'train/val labels size: , {train_labels.size()}, {val_labels.size()}')
        print("Features are ready!\nStart the k-NN classification.")

        knn_with_features(writer, train_embeddings, train_labels, val_embeddings, val_labels, epoch, save_params,
                          if_original, if_eval)
        knn_total_time = time.time() - knn_start_time
        knn_total_time_str = str(datetime.timedelta(seconds=int(knn_total_time)))
        print('KNN time {}'.format(knn_total_time_str))
    return train_embeddings, train_labels, val_embeddings, val_labels


def parse_args(params_path=None):
    parser = argparse.ArgumentParser(description='DINO reimplementation')

    if params_path == None:
        parser.add_argument('--params_path', type=str, required=False,
                            help='Give a valid yaml file that contains all params to load.')
        params_path = parser.parse_args().params_path

    with open(params_path) as f:
        args = yaml.safe_load(f)

    output_dir = args["save_params"]["output_dir"]
    backbone_option = args["model_params"]["backbone_option"]
    dataset_name = args["dataset_params"]["dataset_choice"]["dataset_name"]
    model_name = backbone_option + '_' + dataset_name
    augmentations = args["dataset_params"]['augmentations']
    full_augmentations = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ColorJitter', 'RandomGrayscale', 'GaussianBlur', 'Solarization']
    for each_augmentation in full_augmentations:
        if each_augmentation not in augmentations:
            model_name = model_name + '_' + 'no' + each_augmentation
    if args['start_training']['mode'] == "train_finetuning":
        if args['start_training']['train_finetuning']['ckp_path_choice'] == "Official":
            model_name = model_name + '_finetuning_official'
        elif args['start_training']['train_finetuning']['ckp_path_choice'] == "Ours":
            model_name = model_name + '_finetuning_ours'
        elif args['start_training']['train_finetuning']['ckp_path_choice'] == "Random":
            model_name = model_name + '_random_init'
    model_path = os.path.join(output_dir, model_name)
    print(end='\n\n' + '==' * 50 + '\n\n')
    print('Model is going to be save in ', model_path)
    args["save_params"]["model_path"] = model_path

    print(end='\n\n' + '==' * 50 + '\n\n')
    print('ARGS ARE: ')
    print(json.dumps(args, indent=4))
    return args


def prepare_params(args):
    # Define some parameters for easy access
    start_training = args['start_training']
    save_params = args['save_params']
    dataset_params = args['dataset_params']
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']
    model_params = args['model_params']
    augmentation_params = args['augmentation_params']
    training_params = args['training_params']
    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']
    return start_training, save_params, dataset_params, system_params, dataloader_params, model_params, \
           augmentation_params, training_params, trainloader_params, valloader_params


def prepare_data_model(rank, args):
    start_training, save_params, dataset_params, system_params, dataloader_params, model_params, \
    augmentation_params, training_params, trainloader_params, valloader_params = prepare_params(args)

    # Tensorboard Summarywriter for logging
    # Log network input arg in Tensorboard and save it into a yaml file
    tensorboard_path = os.path.join(save_params['model_path'], 'tensorboard/')
    if rank == 0:
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    writer.add_text("Configuration", json.dumps(args, indent=4))
    output_file_path = os.path.join(save_params['model_path'], 'config.yaml')
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
    with open(output_file_path, 'w') as outfile:
        yaml.dump(args, outfile, sort_keys=False, default_flow_style=False)

    # ============ Preparing system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params['random_seed'])

    # ============ Preparing data ... ============
    # Define transformations applied on augmented and plain data
    # The aug_dataset is for training, while plain_datasets is mostly used to compute knn and visualize the embeddings
    transforms_aug = prepare_datasets.DataAugmentationDINO(dataset_params['augmentations'], augmentation_params['global_crops_scale'],
        augmentation_params['local_crops_scale'], augmentation_params['local_crops_number'],
        augmentation_params['full_size'], augmentation_params['global_size'], augmentation_params['local_size'])
    transforms_plain = transforms_aug.transforms_plain
    transforms_plain_for_lineartrain = transforms_aug.transforms_plain_for_lineartrain
    transforms_plain_for_lineartrain_ddsm = transforms_aug.transforms_plain_for_lineartrain_ddsm

    start_time = time.time()
    dataset_name = dataset_params['dataset_choice']['dataset_name']
    dataset_class = prepare_datasets.GetDatasets(data_folder=dataset_params['data_folder'],
                                                 dataset_name=dataset_name,
                                                 knn_use_cuda=dataset_params['dataset_choice'][dataset_name]['knn_use_cuda'],
                                                 label_mapping_path=dataset_params['dataset_choice'][dataset_name]['label_mapping_path'])
    label_mapping = dataset_class.label_mapping
    use_cuda = dataset_class.use_cuda
    train_aug_dataset = dataset_class.get_datasets('train/', transforms_aug)
    train_plain_dataset = dataset_class.get_datasets('train/', transforms_plain, include_index=True)
    if dataset_name == 'CBISDDSM':
        train_plain_for_lineartrain_dataset = dataset_class.get_datasets('train/', transforms_plain_for_lineartrain_ddsm, include_index=False)
    else:
        train_plain_for_lineartrain_dataset = dataset_class.get_datasets('train/', transforms_plain_for_lineartrain, include_index=False)
    val_plain_dataset = dataset_class.get_datasets('val/', transforms_plain, include_index=True)

    # if train_plain_dataset.classes != val_plain_dataset.classes:
    #     raise ValueError("Inconsistent classes in train and val.")
    if rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f'Building the train_aug_dataset, train_plain_dataset, train_plain_for_lineartrain_dataset and val_plain_dataset took {total_time_str} seconds',
            end='\n\n')

    # Set sampler that restricts data loading to a subset of the dataset
    # In conjunction with torch.nn.parallel.DistributedDataParallel
    train_aug_sampler = torch.utils.data.DistributedSampler(train_aug_dataset, shuffle=True)
    train_plain_sampler = torch.utils.data.DistributedSampler(train_plain_dataset, shuffle=True)
    train_plain_for_lineartrain_sampler = torch.utils.data.DistributedSampler(train_plain_for_lineartrain_dataset, shuffle=True)
    val_plain_sampler = torch.utils.data.DistributedSampler(val_plain_dataset, shuffle=False)

    # Prepare the data for training with DataLoaders
    # pin_memory makes transferring images from CPU to GPU faster
    if start_training['mode'] == 'train':
        train_batch_size = int(trainloader_params['batch_size'] / system_params['num_gpus'])
        val_batch_size = int(valloader_params['batch_size'] / system_params['num_gpus'])
    elif start_training['mode'] == 'eval':
        train_batch_size = int(start_training['eval']['linear']['batch_size'] / system_params['num_gpus'])
        val_batch_size = int(start_training['eval']['linear']['batch_size'] / system_params['num_gpus'])
    elif start_training['mode'] == 'train_finetuning':
        train_batch_size = int(start_training['train_finetuning']['batch_size'] / system_params['num_gpus'])
        val_batch_size = int(start_training['train_finetuning']['batch_size'] / system_params['num_gpus'])

    train_aug_dataloader = DataLoader(train_aug_dataset, sampler=train_aug_sampler,
                                      batch_size=train_batch_size,
                                      num_workers=trainloader_params['num_workers'],
                                      pin_memory=trainloader_params['pin_memory'],
                                      drop_last=trainloader_params['drop_last'])
    train_plain_dataloader = DataLoader(train_plain_dataset, sampler=train_plain_sampler,
                                        batch_size=train_batch_size,
                                        num_workers=trainloader_params['num_workers'],
                                        pin_memory=trainloader_params['pin_memory'],
                                        drop_last=trainloader_params['drop_last'])
    train_plain_for_lineartrain_dataloader = DataLoader(train_plain_for_lineartrain_dataset, sampler=train_plain_for_lineartrain_sampler,
                                        batch_size=train_batch_size,
                                        num_workers=trainloader_params['num_workers'],
                                        pin_memory=trainloader_params['pin_memory'],
                                        drop_last=trainloader_params['drop_last'])
    val_plain_dataloader = DataLoader(val_plain_dataset, sampler=val_plain_sampler,
                                      batch_size=val_batch_size,
                                      num_workers=valloader_params['num_workers'],
                                      pin_memory=valloader_params['pin_memory'],
                                      drop_last=valloader_params['drop_last'])
    if rank == 0:
        print(f"There are {len(train_aug_dataloader)} train_dataloaders on each rank. ")
        print(f"There are {len(train_plain_dataloader)} train_plain_dataloader on each rank. ")
        print(f"There are {len(train_plain_for_lineartrain_dataloader)} train_plain_for_lineartrain_dataloader on each rank. ")
        print(f"There are {len(val_plain_dataloader)} val_plain_dataloader on each rank. ")

    # ============ Building student and teacher networks ... ============
    if rank == 0:
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
    if args['start_training']['mode'] != "eval":
        # Teacher and student start with the same weights and we turn off the gradients on teacher network
        helper.initialize_momentum_state(online_net=student, momentum_net=teacher_without_ddp)

    # Log the number of trainable parameters in Tensorboard
    if rank == 0:
        n_parameters_student = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print('Number of params of the entire student:', n_parameters_student)
        writer.add_text("Number of params of the entire student", str(n_parameters_student))

        n_parameters_student_backbone = sum(p.numel() for p in student.module.backbone.parameters() if p.requires_grad)
        print('Number of params of the student backbone:', n_parameters_student_backbone)
        writer.add_text("Number of params of the student backbone", str(n_parameters_student_backbone))

    return rank, writer, student, teacher, teacher_without_ddp, \
           train_aug_dataloader, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, \
           label_mapping, use_cuda

def main(rank, args):
    # Set up Tensorboard Summarywriter for logging
    # Define system configuration
    # Prepare data, model, loss, optimizer etc

    rank, writer, student, teacher, teacher_without_ddp, \
    train_aug_dataloader, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, \
    label_mapping, use_cuda = prepare_data_model(rank, args)

    print(helper.compute_stats(train_plain_for_lineartrain_dataloader))

if __name__ == '__main__':
    # Read params and print them
    args = parse_args(params_path='yaml/ViT-S-16-CIFAR100.yaml')
    # Launch multi-gpu / distributed training
    helper.launch(main, args)
