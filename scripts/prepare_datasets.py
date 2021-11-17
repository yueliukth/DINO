import os
import json
import shutil
import random
from PIL import Image
from PIL import ImageFilter, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.distributed as dist
from fastai.vision.all import *

import helper

class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, lab, idx

class GetDatasets():
    def __init__(self, dataset_params):
        self.dataset_params = dataset_params
        self.data_folder = dataset_params['data_folder']
        self.dataset_name = dataset_params['specification']['dataset_name']
        self.use_cuda = dataset_params['specification'][self.dataset_name]['knn_use_cuda']
        label_mapping_path = dataset_params['specification'][self.dataset_name]['label_mapping_path']
        f = open(label_mapping_path)
        self.label_mapping = json.load(f)

    def get_datasets(self, official_split, transforms, include_index=False):
        #transforms = ToTensor()
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

        # elif self.dataset_name == 'FashionMNIST':
        #     # Train: 60,000 images
        #     # Test: 10,000 images
        #     # We consider the test set of FashionMNIST as validation set in our task
        #     if official_split == 'train':
        #         dataset = datasets.FashionMNIST(
        #         root=self.data_folder,
        #         train=True,
        #         download=True,
        #         transform=transforms)
        #     else:
        #         dataset = datasets.FashionMNIST(
        #         root=self.data_folder,
        #         train=False,
        #         download=True,
        #         transform=transforms)
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
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, full_size, global_size, local_size):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])


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

        # first global crop
        self.global_transforms1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transforms2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transforms1(image))
        crops.append(self.global_transforms2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transforms(image))
        return crops

