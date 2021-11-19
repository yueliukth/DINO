import os
import json
import shutil
import random
import tarfile
import pandas as pd
from PIL import Image
from PIL import ImageFilter, ImageOps

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
import torch.distributed as dist
#from fastai.vision.all import *

import helper

class FlowerDataset(Dataset):
    def __init__(self, img_folder, label_path_list, transforms=None, include_index=False):
        """
        Args:
            img_folder (string): Directory with all the images.
            label_path_list (string): Path to the csv files with labels.
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        """

        label_csv_list = []
        for label_path in label_path_list:
            label_csv = pd.read_csv(label_path, delimiter=' ')
            label_csv_list.append(label_csv)
        self.label_csv_combined = pd.concat(label_csv_list, ignore_index=True)
        self.img_folder = img_folder
        self.transforms = transforms
        self.include_index = include_index
    def __len__(self):
        return len(self.label_csv_combined)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.label_csv_combined.iloc[idx, 0])
        lab = torch.tensor(self.label_csv_combined.iloc[idx,1])
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        if self.include_index:
            return img, lab, idx
        else:
            return img, lab

class CBISDDSMDataset(Dataset):
    def __init__(self, img_folder, csv_path, official_split, transforms=None, include_index=False):
        """
        Args:
            img_folder (string): Directory with all the images.
            csv_path (string): Path to the csv file with labels.
            official_split (string): 'train/' or 'val/'
            transforms (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.img_folder = img_folder
        csv = pd.read_csv(csv_path)
        if 'train' in official_split:
            self.label_csv = csv[csv['split']=='train'].reset_index(drop=True)
        elif 'val' in official_split:
            self.label_csv = csv[csv['split']=='val'].reset_index(drop=True)

        self.official_split = official_split
        self.transforms = transforms
        self.include_index = include_index
    def __len__(self):
        return len(self.label_csv)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.label_csv.iloc[idx, 'img_path'])
        lab = torch.tensor(self.label_csv.iloc[idx, 'label'])
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        if self.include_index:
            return img, lab, idx
        else:
            return img, lab

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab, idx

class ReturnIndexDatasetCIFAR10(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDatasetCIFAR10, self).__getitem__(idx)
        return img, lab, idx

class ReturnIndexDatasetCIFAR100(datasets.CIFAR100):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDatasetCIFAR100, self).__getitem__(idx)
        return img, lab, idx

class GetDatasets():
    def __init__(self, data_folder, dataset_name, knn_use_cuda, label_mapping_path):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.use_cuda = knn_use_cuda
        if self.dataset_name == 'CBISDDSM':
            self.label_mapping = {'0': 'calcification', '1': 'mass'}
        else:
            if os.path.exists(label_mapping_path):
                f = open(label_mapping_path)
                self.label_mapping = json.load(f)
            else:
                self.label_mapping = None

    def get_datasets(self, official_split, transforms, include_index=False):
        #transforms = ToTensor()
        if self.dataset_name in ['ImageNet', 'IMAGENETTE', 'Flower']:
            if self.dataset_name == 'ImageNet':
                # Train: 1,281,167 images
                # Val: 50,000 images
                # Test: 100,000 images
                target_path = os.path.join(self.data_folder, 'imagenet/ILSVRC/Data/CLS-LOC/')
            elif self.dataset_name == 'IMAGENETTE':
                # Train: 9,469 images
                # Val: 3,925 images
                target_path = os.path.join(self.data_folder, "imagenette2")
                target_path = os.path.abspath(target_path)
                if not os.path.exists(target_path) or len(os.listdir(target_path)) == 0:
                    # If IMAGENETTE does not exist in data folder, we download it with fastai
                    path = untar_data(URLs.IMAGENETTE)
                    if not os.path.exists(os.path.dirname(target_path)):
                        os.makedirs(os.path.dirname(target_path))
                    # Copy data folder from .fastai/data to our local DINO folder
                    shutil.copytree(path, target_path)
            if include_index:
                dataset = ReturnIndexDataset(os.path.join(target_path, official_split), transform=transforms)
            else:
                dataset = ImageFolder(os.path.join(target_path, official_split), transform=transforms)

        elif self.dataset_name in ['Flower', 'CBISDDSM']:
            if self.dataset_name == 'Flower':
                # Number of classes: 102
                # Each class consists of between 40 and 258 images
                # Train: 1020 images (10 images per class)
                # Val: 1020 images (10 images per class)
                # Test: the remaining 6149 images (minimum 20 per class)
                target_path = os.path.join(self.data_folder, "Flower")
                target_path = os.path.abspath(target_path)
                if not os.path.exists(target_path) or len(os.listdir(target_path)) == 0:
                    # If Flower does not exist in data folder, we download it
                    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"
                    download_url(dataset_url, target_path)
                    # Extract from archive
                    with tarfile.open(os.path.join(target_path, 'oxford-102-flowers.tgz'), 'r:gz') as tar:
                        tar.extractall(path=target_path)

                img_folder = os.path.join(target_path, 'oxford-102-flowers')
                if 'train' in official_split:
                    label_path_list = [os.path.join(img_folder, 'jpg', 'train.txt'),
                                   os.path.join(img_folder, 'jpg', 'valid.txt')]
                elif 'val' in official_split:
                    label_path_list = [os.path.join(img_folder, 'jpg', 'test.txt')]

                dataset = FlowerDataset(img_folder=img_folder, label_path_list=label_path_list, transforms=transforms, include_index=include_index)
            elif self.dataset_name == 'CBISDDSM':
                # Number of classes: 2 {'0': 'calcification'-1511 images, '1': 'mass'-1592 images}
                # Number of images: 3103 images from 1566 patients
                # Data split on patient-level 80% train, 20% val, resulting 1252 train patients and 314 val patients
                # Train: 2484 images ('0' - 1203 images, '1' - 1281 images)
                # Val: 619 images ('0' -308 images, '1' - 311 images)
                target_path = os.path.join(self.data_folder, "CBISDDSM")
                img_folder = os.path.join(target_path, 'imgs')
                csv_path = os.path.join(target_path, 'ddsm_csv_for_dataset.csv')
                dataset = CBISDDSMDataset(img_folder=img_folder, csv_path=csv_path, official_split=official_split, transforms=transforms, include_index=include_index)

        elif self.dataset_name in ['CIFAR10', 'CIFAR100']:
            if self.dataset_name == 'CIFAR10':
                # Train: 50,000 images
                # Test: 10,000 images
                # We consider the test set of CIFAR10 as validation set in our task
                if include_index:
                    dataset = ReturnIndexDatasetCIFAR10(
                        root=self.data_folder,
                        train=(official_split == 'train/'),
                        download=True,
                        transform=transforms)
                else:
                    dataset = datasets.CIFAR10(
                        root=self.data_folder,
                        train=(official_split == 'train/'),
                        download=True,
                        transform=transforms)
            elif self.dataset_name == 'CIFAR100':
                # Train: 50,000 images
                # Test: 10,000 images
                # We consider the test set of CIFAR100 as validation set in our task
                if include_index:
                    dataset = ReturnIndexDatasetCIFAR100(
                        root=self.data_folder,
                        train=(official_split == 'train/'),
                        download=True,
                        transform=transforms)
                else:
                    dataset = datasets.CIFAR100(
                        root=self.data_folder,
                        train=(official_split == 'train/'),
                        download=True,
                        transform=transforms)

        if helper.is_main_process():
            print(f"There are {len(dataset)} images in {official_split} split, on each rank. ")
        return dataset


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DataAugmentationDINO(object):
    # Adopted from the original DINO implementation
    # Removed the hardcoded global_size and local_size
    """Create crops of an input image together with additional augmentation.
    It generates 2 global crops and `n_local_crops` local crops.
    Parameters
    ----------
    n_chnl: int
        Number of channels in inputimages
    global_crops_scale : list
        Range of sizes for the global crops.
    local_crops_scale : list
        Range of sizes for the local crops.
    local_crops_number : int
        Number of local crops to create.
    full_size: int
        The size of the full image, eg, on ImageNet, the standard setting is 256.
    global_size : int
        The size of the final global crop.
    local_size : int
        The size of the final local crop.
    Attributes
    ----------
    global_transforms1, global_transforms2 : transforms.Compose
        Two global transforms.
    local_transforms : transforms.Compose
        Local transform. Note that the augmentation is stochastic so one
        instance is enough and will lead to different crops.
    """
    def __init__(self, augmentations, global_crops_scale, local_crops_scale, local_crops_number, full_size, global_size, local_size):
        normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        self.transforms_plain = transforms.Compose([
            transforms.Resize(full_size, interpolation=3),
            transforms.CenterCrop(global_size),
            normalize])
        self.transforms_plain_for_lineartrain = transforms.Compose([
            transforms.RandomResizedCrop(global_size),
            transforms.RandomHorizontalFlip(),
            normalize])
        global_transforms1_list = []
        global_transforms2_list = []
        local_transforms_list = []

        if 'RandomResizedCrop' in augmentations:
            global_transforms1_list.append(transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC))
            global_transforms2_list.append(transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC))
            local_transforms_list.append(transforms.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=Image.BICUBIC))
        else:
            global_transforms1_list.append(transforms.RandomResizedCrop(global_size))
            global_transforms2_list.append(transforms.RandomResizedCrop(global_size))
            local_transforms_list.append(transforms.RandomResizedCrop(local_size))

        if 'RandomHorizontalFlip' in augmentations:
            global_transforms1_list.append(transforms.RandomHorizontalFlip(p=0.5))
            global_transforms2_list.append(transforms.RandomHorizontalFlip(p=0.5))
            local_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5),)

        if 'ColorJitter' in augmentations:
            global_transforms1_list.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8))
            global_transforms2_list.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8))
            local_transforms_list.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8))

        if 'RandomGrayscale' in augmentations:
            global_transforms1_list.append(transforms.RandomGrayscale(p=0.2))
            global_transforms2_list.append(transforms.RandomGrayscale(p=0.2))
            local_transforms_list.append(transforms.RandomGrayscale(p=0.2))

        if 'GaussianBlur' in augmentations:
            global_transforms1_list.append(GaussianBlur(p=1.0))
            global_transforms2_list.append(GaussianBlur(p=0.1))
            local_transforms_list.append(GaussianBlur(p=0.5))

        if 'Solarization' in augmentations:
            global_transforms2_list.append(Solarization(0.2))

        global_transforms1_list.append(normalize)
        global_transforms2_list.append(normalize)
        local_transforms_list.append(normalize)

        # first global crop
        self.global_transforms1 = transforms.Compose(global_transforms1_list)
        # second global crop
        self.global_transforms2 = transforms.Compose(global_transforms2_list)
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transforms = transforms.Compose(local_transforms_list)

    def __call__(self, image):
        crops = []
        crops.append(self.global_transforms1(image))
        crops.append(self.global_transforms2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transforms(image))
        return crops

