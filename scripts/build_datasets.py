import os
import shutil
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from fastai.vision.all import *

def get_train_test_data(dataset_name, transforms):
    transforms = ToTensor()
    if dataset_name == 'ImageNet':
        return None
    elif dataset_name == 'FashionMNIST':
        training_data = datasets.FashionMNIST(
            root="../data",
            train=True,
            download=True,
            transform=transforms)
        test_data = datasets.FashionMNIST(
            root="../data",
            train=False,
            download=True,
            transform=transforms)
    elif dataset_name == 'IMAGENETTE':
        target_path = "../data/imagenette2"
        target_path = os.path.abspath(target_path)
        if not os.path.exists(target_path) or len(os.listdir(target_path)) == 0:
            # If IMAGENETTE does not exist in data folder, we download it with fastai
            path = untar_data(URLs.IMAGENETTE)
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))
            # Copy data folder from .fastai/data to our local DINO folder
            shutil.copytree(path, target_path)
        training_data = ImageFolder(os.path.join(target_path, 'train'), transform=transforms)
        test_data = ImageFolder(os.path.join(target_path, 'val'), transform=transforms)

    return training_data, test_data