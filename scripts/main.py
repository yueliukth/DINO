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

    print(end='\n\n' + '==' * 50 + '\n\n')
    print('ARGS ARE: ')
    print(json.dumps(args, indent=4))
    return args


def prepare_params(args):
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
    return save_params, dataset_params, system_params, dataloader_params, model_params, \
           augmentation_params, training_params, trainloader_params, valloader_params


def prepare_data_model(rank, args):
    save_params, dataset_params, system_params, dataloader_params, model_params, \
    augmentation_params, training_params, trainloader_params, valloader_params = prepare_params(args)

    # Tensorboard Summarywriter for logging
    # Log network input arg in Tensorboard and save it into a yaml file
    tensorboard_path = os.path.join(save_params['output_dir'], 'tensorboard/')
    if rank == 0:
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)
    writer.add_text("Configuration", json.dumps(args, indent=4))
    output_file_path = os.path.join(save_params['output_dir'], 'config.yaml')
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))
    with open(output_file_path, 'w') as outfile:
        yaml.dump(args, outfile, sort_keys=False, default_flow_style=False)

    # ============ Preparing system configuration ... ============
    # Set gpu params and random seeds for reproducibility
    helper.set_sys_params(system_params)

    # ============ Preparing data ... ============
    # Define transformations applied on augmented and plain data
    # The aug_dataset is for training, while plain_datasets is mostly used to compute knn and visualize the embeddings
    transforms_aug = prepare_datasets.DataAugmentationDINO(augmentation_params['global_crops_scale'],
        augmentation_params['local_crops_scale'], augmentation_params['local_crops_number'],
        augmentation_params['full_size'], augmentation_params['global_size'], augmentation_params['local_size'])
    transforms_plain = transforms_aug.transforms_plain
    transforms_plain_for_lineartrain = transforms_aug.transforms_plain_for_lineartrain

    start_time = time.time()
    label_mapping = prepare_datasets.GetDatasets(dataset_params).label_mapping
    train_aug_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('train/', transforms_aug)
    train_plain_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('train/', transforms_plain,
                                                                                    include_index=True)
    train_plain_for_lineartrain_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('train/', transforms_plain_for_lineartrain,
                                                                                    include_index=False)
    val_plain_dataset = prepare_datasets.GetDatasets(dataset_params).get_datasets('val/', transforms_plain,
                                                                                  include_index=True)

    if train_plain_dataset.classes != val_plain_dataset.classes:
        raise ValueError("Inconsistent classes in train and val.")
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
    train_plain_for_lineartrain_dataloader = DataLoader(train_plain_for_lineartrain_dataset, sampler=train_plain_for_lineartrain_sampler,
                                        batch_size=int(trainloader_params['batch_size'] / system_params['num_gpus']),
                                        num_workers=trainloader_params['num_workers'],
                                        pin_memory=trainloader_params['pin_memory'],
                                        drop_last=trainloader_params['drop_last'])
    val_plain_dataloader = DataLoader(val_plain_dataset, sampler=val_plain_sampler,
                                      batch_size=int(valloader_params['batch_size'] / system_params['num_gpus']),
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

    if dataset_params['dataset_name'] == 'ImageNet':
        use_cuda = False
    else:
        use_cuda = False

    return rank, writer, student, teacher, teacher_without_ddp, \
           train_aug_dataloader, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, \
           train_aug_dataset, label_mapping, use_cuda


def train_process(rank, args, writer, student, teacher, teacher_without_ddp, train_aug_dataloader,
                  train_plain_dataloader, val_plain_dataloader, train_aug_dataset, label_mapping, use_cuda):

    save_params, dataset_params, system_params, dataloader_params, model_params, \
    augmentation_params, training_params, trainloader_params, valloader_params = prepare_params(args)

    # ============ Preparing loss ... ============
    # Move dino_loss to gpu
    dino_loss = prepare_losses.DINOLoss(out_dim=model_params['out_dim'],
                                        num_crops=augmentation_params['local_crops_number'] + 2,
                                        warmup_teacher_temp=training_params['warmup_teacher_temp'],
                                        teacher_temp=training_params['teacher_temp'],
                                        warmup_teacher_temp_epochs=training_params['warmup_teacher_temp_epochs'],
                                        num_epochs=training_params['num_epochs'],
                                        student_temp=training_params['student_temp'],
                                        center_momentum=training_params['center_momentum']).cuda()

    # ============ Preparing optimizer ... ============
    optimizer = prepare_trainers.get_optimizer(optimizer_choice=training_params['optimizer']['name'],
                                               params_dict=helper.get_params_groups(student),
                                               lr=training_params['optimizer']['sgd']['lr'],
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
    base_lr = training_params['lr']['base_lr'] * trainloader_params['batch_size_for_scheduler'] / 256.
    lr_schedule = helper.cosine_scheduler(base_value=base_lr, final_value=training_params['lr']['final_lr'],
                                          epochs=training_params['num_epochs_for_scheduler'], niter_per_ep=len(train_aug_dataset),
                                          warmup_epochs=training_params['lr']['warmup_epochs'],
                                          start_warmup_value=training_params['lr']['start_warmup_lr'], )

    if rank == 0:
        print()
        print('base_lr: ', base_lr)
        print('final_value: ', training_params['lr']['final_lr'])
        print('epochs: ', training_params['num_epochs_for_scheduler'])
        print('niter_per_ep: ', niter_per_ep)
        print('warmup_epochs: ', training_params['lr']['warmup_epochs'])
        print('start_warmup_value: ', training_params['lr']['start_warmup_lr'],)
        print('Learning rate at epoch 9 should be :', lr_schedule[9*len(train_aug_dataset)], end='\n\n')

    wd_schedule = helper.cosine_scheduler(base_value=training_params['wd']['base_wd'],
                                          final_value=training_params['wd']['final_wd'],
                                          epochs=training_params['num_epochs_for_scheduler'], niter_per_ep=len(train_aug_dataset), )
    # Momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = helper.cosine_scheduler(base_value=training_params['momentum']['base_momentum_teacher'],
                                                final_value=training_params['momentum']['final_momentum_teacher'],
                                                epochs=training_params['num_epochs_for_scheduler'],
                                                niter_per_ep=len(train_aug_dataset))
    if rank == 0:
        print(f"Loss, optimizer and schedulers ready.")

    # ============ Optionally resume training ... ============
    to_restore = {'epoch': save_params['restore_epoch']}
    helper.restart_from_checkpoint(os.path.join(save_params['output_dir'], "checkpoint.pth"), run_variables=to_restore,
                                   student=student, teacher=teacher, optimizer=optimizer, fp16_scaler=fp16_scaler,
                                   dino_loss=dino_loss, )
    print("Starting DINO training !")
    # Record the starting time
    start_time = time.time()
    if save_params['tb_logoriginal']:
        # ============ Adding embeddings in tensorboard before the training starts ... ============
        tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(teacher_without_ddp.backbone,
                                                                                 val_plain_dataloader, label_mapping,
                                                                                 use_cuda=use_cuda, return_tb=True,
                                                                                 subset_size=100)
        if rank == 0:
            writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img,
                                 global_step=to_restore['epoch'], tag="embeddings_original")

        # ============ Adding knn results in tensorboard before the training starts ... ============
        _, _, _, _ = knn_in_train_process(rank=rank, writer=writer, use_cuda=use_cuda, backbone=teacher_without_ddp.backbone,
                                          train_dataloader=train_plain_dataloader, val_dataloader=val_plain_dataloader,
                                          label_mapping=label_mapping, epoch=to_restore['epoch'],
                                          save_params=save_params, if_original=True)

    # ============ Start training ... ============
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
        if rank == 0:
            print('Training one epoch is done, start writting loss and learning rate in tensorboard...')
            writer.add_scalar("train_loss", train_stats['loss'], epoch)
            print(f"Training learing rate at epoch {epoch} is : {train_stats['lr']}")
            writer.add_scalar("train_lr", train_stats['lr'], epoch)
            print(f"Training weight decay at epoch {epoch} is : {train_stats['wd']}")
            writer.add_scalar("train_wd", train_stats['wd'], epoch)



        # Log the embeddings & KNN results in Tensorboard, at every tb_freq epoch
        if epoch % save_params['tb_freq'] == 0:
            tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(teacher_without_ddp.backbone,
                                                                                     val_plain_dataloader,
                                                                                     label_mapping, return_tb=True,
                                                                                     subset_size=100)
            if rank == 0:
                writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img, global_step=epoch,
                                     tag="embeddings")

            train_embeddings, train_labels, val_embeddings, val_labels = knn_in_train_process(rank=rank, writer=writer,
                                                                                              use_cuda=use_cuda,
                                                                                              backbone=teacher_without_ddp.backbone,
                                                                                              train_dataloader=train_plain_dataloader,
                                                                                              val_dataloader=val_plain_dataloader,
                                                                                              label_mapping=label_mapping,
                                                                                              epoch=epoch,
                                                                                              save_params=save_params)
            # If the current epoch is the last epoch, we save the embeddings
            if rank == 0 and epoch == training_params['num_epochs'] - 1:
                torch.save(train_embeddings.cpu(),
                           os.path.join(save_params['output_dir'], f"trainembeddings{epoch:04}.pth"))
                torch.save(val_embeddings.cpu(),
                           os.path.join(save_params['output_dir'], f"valembeddings{epoch:04}.pth"))
                torch.save(train_labels.cpu(), os.path.join(save_params['output_dir'], f"trainlabels{epoch:04}.pth"))
                torch.save(val_labels.cpu(), os.path.join(save_params['output_dir'], f"vallabels{epoch:04}.pth"))

        # ============ Saving models and writing logs in log.txt file ... ============
        save_dict = {'student': student.state_dict(), 'teacher': teacher.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'args': args,
                     'dino_loss': dino_loss.state_dict(), }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        if rank == 0:
            torch.save(save_dict, os.path.join(save_params['output_dir'], "checkpoint.pth"))
            if save_params['saveckp_freq'] and epoch % save_params['saveckp_freq'] == 0:
                torch.save(save_dict, os.path.join(save_params['output_dir'], f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if rank == 0:
            with (Path(save_params['output_dir']) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    writer.close()
    # Log the number of total training time in Tensorboard
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if rank == 0:
        with (Path(save_params['output_dir']) / "log.txt").open("a") as f:
            f.write(
                f"Training time from epoch {to_restore['epoch']} to epoch {training_params['num_epochs']}: " + total_time_str + "\n")


def eval_linear(rank, writer, train_plain_for_lineartrain_dataloader, val_plain_dataloader, teacher_without_ddp, start_training, dataset_params, model_params, save_params):
    if rank == 0:
        print("Start linear evaluation...")

    eval_linear = start_training['eval']['linear']
    # ============ Building model with linear classifier ... ============
    embed_dim = teacher_without_ddp.backbone.embed_dim * (eval_linear['n_last_blocks'] + int(eval_linear['avgpool_patchtokens']))
    if dataset_params['dataset_name'] == 'ImageNet':
        linear_classifier = evaluation.LinearClassifier(embed_dim, num_labels=1000)
    elif dataset_params['dataset_name'] == 'IMAGENETTE':
        linear_classifier = evaluation.LinearClassifier(embed_dim, num_labels=10)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[rank])

    # Log the number of trainable parameters in Tensorboard
    if rank == 0:
        n_parameters_teacher_without_ddp = sum(p.numel() for p in teacher_without_ddp.backbone.parameters() if p.requires_grad)
        print('Linear evaluation: Number of params of the teacher backbone:', n_parameters_teacher_without_ddp)
        writer.add_text("Linear evaluation:  Number of params of the teacher backbone", str(n_parameters_teacher_without_ddp))

        n_parameters_linear = sum(p.numel() for p in linear_classifier.parameters() if p.requires_grad)
        print('Linear evaluation: Number of params of the linear classifier:', n_parameters_linear)
        writer.add_text("Linear evaluation:  Number of params of the linear classifier", str(n_parameters_linear))

    # ============ Preparing optimizer ... ============
    optimizer_linear = torch.optim.SGD(
        linear_classifier.parameters(),
        eval_linear['lr'] * eval_linear['batch_size']/ 256., # linear scaling rule
        momentum=eval_linear['momentum'],
        weight_decay=eval_linear['wd'],
        )
    scheduler_linear = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_linear, eval_linear['num_epochs'])

    # ============ Optionally resume training ... ============
    to_restore_linear = {'epoch': eval_linear['restore_epoch'], "best_acc": 0.}
    helper.restart_from_checkpoint(os.path.join(save_params['output_dir'], "checkpoint_linear.pth"),
                                   run_variables=to_restore_linear,
                                   state_dict=linear_classifier,
                                   optimizer=optimizer_linear,
                                   scheduler=scheduler_linear)
    best_acc = to_restore_linear["best_acc"]

    # ============ Start training ... ============
    for epoch in range(to_restore_linear["epoch"], eval_linear['num_epochs']):
        train_plain_for_lineartrain_dataloader.sampler.set_epoch(epoch)
        val_plain_dataloader.sampler.set_epoch(epoch)

        # ============ Training one epoch of model with linear classifier ... ============
        train_stats_linear = prepare_trainers.linear_train_one_epoch(teacher_without_ddp.backbone, linear_classifier, optimizer_linear, train_plain_for_lineartrain_dataloader, epoch, \
                                                                     eval_linear['n_last_blocks'], eval_linear['avgpool_patchtokens'], \
                                                                     model_params)
        scheduler_linear.step()

        # ============ Writing logs in tensorboard ... ============
        # Log the number of training loss with linear classifier in Tensorboard, at every epoch
        if rank == 0:
            writer.add_scalar("train_loss_linear", train_stats_linear['loss'], epoch)

        log_stats_linear = {**{f'train_{k}': v for k, v in train_stats_linear.items()}, 'epoch': epoch}

        if epoch % eval_linear['val_freq'] == 0 or epoch == eval_linear['num_epochs'] - 1:
            val_stats_linear = evaluation.validate_network(val_plain_dataloader, teacher_without_ddp.backbone, linear_classifier, eval_linear['n_last_blocks'], eval_linear['avgpool_patchtokens'], model_params)
            print(f"Accuracy at epoch {epoch} of the network on the validation set: {val_stats_linear['acc1']:.1f}%")
            best_acc = max(best_acc,val_stats_linear['acc1'])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats_linear = {**{k: v for k, v in log_stats_linear.items()},
                                **{f'val_{k}': v for k, v in val_stats_linear.items()}}
            if rank == 0:
                writer.add_scalar("train_acc1_linear", val_stats_linear['acc1'], epoch)
                writer.add_scalar("train_max_acc1_linear", best_acc, epoch)

        # ============ Saving models and writing logs in log.txt file ... ============
        save_dict_linear = {"epoch": epoch + 1, "state_dict": linear_classifier.state_dict(),
                            "optimizer": optimizer_linear.state_dict(), "scheduler": scheduler_linear.state_dict(), "best_acc": best_acc}
        if rank == 0:
            torch.save(save_dict_linear, os.path.join(save_params['output_dir'], f'checkpoint{epoch:04}_linear.pth'))
            with (Path(save_params['output_dir']) / "log_linear.txt").open("a") as f:
                f.write(json.dumps(log_stats_linear) + "\n")
    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    return writer

    
def eval_process(rank, args, writer, teacher_without_ddp, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, \
                 label_mapping, use_cuda):
    start_training = args['start_training']
    save_params, dataset_params, system_params, dataloader_params, model_params, \
    augmentation_params, training_params, trainloader_params, valloader_params = prepare_params(args)

    epoch = int(start_training['eval']['epoch'])
    print("Starting DINO evaluation at epoch ", str(epoch))

    # Record the starting time
    start_time = time.time()
    checkpoint = torch.load(os.path.join(save_params['output_dir'], f"checkpoint{epoch:04}.pth"))
    # Set the teacher model as inference (evaluating) time
    teacher_without_ddp.eval()
    teacher_without_ddp.load_state_dict(checkpoint['teacher'])

    if 'if_linear' in start_training['eval']['choices']:
        writer = eval_linear(rank, writer, train_plain_for_lineartrain_dataloader, val_plain_dataloader, teacher_without_ddp, start_training, dataset_params, model_params, save_params)

    if 'if_knn' in start_training['eval']['choices']:
        if rank == 0:
            print("Adding knn results in tensorboard...")
        trainembeddings_path = os.path.join(save_params['output_dir'], f"trainembeddings{epoch:04}.pth")
        valembeddings_path = os.path.join(save_params['output_dir'], f"valembeddings{epoch:04}.pth")
        trainlabels_path = os.path.join(save_params['output_dir'], f"trainlabels{epoch:04}.pth")
        vallabels_path = os.path.join(save_params['output_dir'], f"vallabels{epoch:04}.pth")
        if os.path.exists(trainembeddings_path) and \
                os.path.exists(valembeddings_path) and \
                os.path.exists(trainlabels_path) and \
                os.path.exists(vallabels_path):
            if rank == 0:
                print('Embeddings and labels already exist, loading them ....')
            train_embeddings = torch.load(trainembeddings_path)
            val_embeddings = torch.load(valembeddings_path)
            train_labels = torch.load(trainlabels_path)
            val_labels = torch.load(vallabels_path)
            if rank==0:
                knn_with_features(writer=writer, train_embeddings=train_embeddings, train_labels=train_labels, \
                                  val_embeddings=val_embeddings, val_labels=val_labels, epoch=epoch, save_params=save_params, if_original=False, if_eval=True)
        else:
            print('Calculating embeddings and labels...')
            train_embeddings, train_labels, val_embeddings, val_labels = \
                knn_in_train_process(rank=rank, writer=writer,
                                     use_cuda=use_cuda,
                                     backbone=teacher_without_ddp.backbone,
                                     train_dataloader=train_plain_dataloader,
                                     val_dataloader=val_plain_dataloader,
                                     label_mapping=label_mapping,
                                     epoch=epoch,
                                     save_params=save_params,
                                     if_eval=True)

            if rank == 0:
                print('Saving the calculated embeddings and labels...')
                torch.save(train_embeddings.cpu(), trainembeddings_path)
                torch.save(val_embeddings.cpu(), valembeddings_path)
                torch.save(train_labels.cpu(), trainlabels_path)
                torch.save(val_labels.cpu(), vallabels_path)

    if 'if_embeddings' in start_training['eval']['choices']:
        if rank == 0:
            print("Adding embeddings in tensorboard...")
        tb_embeddings, tb_label_img, tb_metatdata = evaluation.compute_embedding(teacher_without_ddp.backbone,
                                                                                 val_plain_dataloader, label_mapping,
                                                                                 return_tb=True, subset_size=100)
        if rank == 0:
            writer.add_embedding(tb_embeddings, metadata=tb_metatdata, label_img=tb_label_img, global_step=epoch,
                                 tag="embeddings_eval")

    if 'if_throughput' in start_training['eval']['choices']:
        print("Calculating throughput...")
        assert system_params['gpu_ids'] == "0"
        assert valloader_params['batch_size'] == 128

        device = torch.device("cuda:0")
        teacher_without_ddp.to(device)
        with torch.no_grad():
            for img, _, _ in val_plain_dataloader:
                break
            total_time = 0
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = teacher_without_ddp(img)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
        throughput = 128 / total_time
        print('Final Throughput: ', throughput)

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if rank == 0:
        print('Evaluation time {}'.format(total_time_str))

def main(rank, args):
    # Set up Tensorboard Summarywriter for logging
    # Define system configuration
    # Prepare data, model, loss, optimizer etc
    rank, writer, student, teacher, teacher_without_ddp,\
    train_aug_dataloader, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, train_aug_dataset, label_mapping, use_cuda = prepare_data_model(rank, args)

    if args['start_training']['mode'] == "train":
        # Start training
        train_process(rank, args, writer, student, teacher, teacher_without_ddp, train_aug_dataloader,
                      train_plain_dataloader, val_plain_dataloader, train_aug_dataset, label_mapping, use_cuda)
    elif args['start_training']['mode'] == "eval":
        # Start evaluation
        eval_process(rank, args, writer, teacher_without_ddp, train_plain_dataloader, train_plain_for_lineartrain_dataloader, val_plain_dataloader, label_mapping, use_cuda)


if __name__ == '__main__':
    # Read params and print them
    # args = parse_args(params_path='yaml/ViT-S-16.yaml')
    args = parse_args(params_path='yaml/ViT-S-16-imagenette.yaml')
    # args = parse_args(params_path='yaml/ResNet50.yaml')

    # Launch multi-gpu / distributed training
    helper.launch(main, args)
