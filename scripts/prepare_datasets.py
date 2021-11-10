import os
import shutil
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.distributed as dist
from fastai.vision.all import *

def get_datasets(dataset_params, official_split, transforms):
    #transforms = ToTensor()
    if dataset_params['dataset_name'] == 'ImageNet':
        # Train: 1,281,167 images
        # Val: 50,000 images
        # Test: 100,000 images
        target_path = os.path.join(dataset_params['data_folder'], 'imagenet/ILSVRC/Data/CLS-LOC/')
        dataset = ImageFolder(os.path.join(target_path, official_split), transform=transforms)
    elif dataset_params['dataset_name'] == 'IMAGENETTE':
        # Train: 9,469 images
        # Val: 3,925 images
        target_path = os.path.join(dataset_params['data_folder'], "imagenette2")
        target_path = os.path.abspath(target_path)
        if not os.path.exists(target_path) or len(os.listdir(target_path)) == 0:
            # If IMAGENETTE does not exist in data folder, we download it with fastai
            path = untar_data(URLs.IMAGENETTE)
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            # Copy data folder from .fastai/data to our local DINO folder
            shutil.copytree(path, target_path)
        dataset = ImageFolder(os.path.join(target_path, official_split), transform=transforms)
    elif dataset_params['dataset_name'] == 'FashionMNIST':
        # Train: 60,000 images
        # Test: 10,000 images
        # We consider the test set of FashionMNIST as validation set in our task
        if official_split == 'train':
            dataset = datasets.FashionMNIST(
            root=dataset_params['data_folder'],
            train=True,
            download=True,
            transform=transforms)
        else:
            dataset = datasets.FashionMNIST(
            root=dataset_params['data_folder'],
            train=False,
            download=True,
            transform=transforms)
    return dataset