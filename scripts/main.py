import os
import sys
import yaml
import argparse

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import helper

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

def train_process(args, start_training=True):
    system_params = args['system_params']
    dataloader_params = args['dataloader_params']

    print('Start preparing for training ')
    # TODO: Set up data loader with augmentations
    # Load an example Dataset with distributed training, for debugging purpose
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Set sampler that restricts data loading to a subset of the dataset
    # In conjunction with torch.nn.parallel.DistributedDataParallel
    training_sampler = torch.utils.data.DistributedSampler(training_data, shuffle=True)
    test_sampler = torch.utils.data.DistributedSampler(test_data, shuffle=True)

    # Preparing the data for training with DataLoaders
    trainloader_params = dataloader_params['trainloader']
    valloader_params = dataloader_params['valloader']
    train_dataloader = DataLoader(training_data, batch_size=int(trainloader_params['batch_size']/system_params['num_gpus']),
                                  num_workers=trainloader_params['num_workers'], pin_memory=trainloader_params['pin_memory'], drop_last=trainloader_params['drop_last'])
    test_dataloader = DataLoader(test_data, batch_size=int(valloader_params['batch_size']/system_params['num_gpus']),
                                 num_workers=valloader_params['num_workers'], pin_memory=valloader_params['pin_memory'], drop_last=valloader_params['drop_last'])
    print(f"Data loaded: there are {len(training_data)} training images.")
    print(f"Data loaded: there are {len(test_data)} test images.")

    # TODO: Build the student and teacher networks

    # TODO: Set up the training procedure

    return

def main(args):
    # Set up training
    train_process(args, start_training=True)

if __name__ == '__main__':
    # Read params and print them
    args = parse_args(params_path='yaml/test_params.yaml')
    # Launch multi-gpu or distributed training
    helper.launch(main, args)