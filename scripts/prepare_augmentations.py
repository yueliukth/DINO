import random
from PIL import Image
from PIL import ImageFilter, ImageOps
import torch
from torchvision import transforms

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
    def __init__(self, dataset_params, global_crops_scale, local_crops_scale, local_crops_number, full_size, global_size, local_size):
        dataset_name = dataset_params['dataset_choice']['dataset_name']
        num_channels = dataset_params['dataset_choice'][dataset_name]['num_channels']
        if num_channels == 3:
            normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        elif num_channels == 1:
            normalize = transforms.Compose([transforms.Normalize((0.286, 0.286, 0.286), (0.267, 0.267, 0.267)),])
        augmentations = dataset_params['augmentations']
    
        self.transforms_plain = transforms.Compose([
            transforms.Resize(full_size, interpolation=3),
            transforms.CenterCrop(global_size),
            transforms.ToTensor(),
            normalize])
        self.transforms_plain_for_lineartrain = transforms.Compose([
            transforms.RandomResizedCrop(global_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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

        global_transforms1_list.extend([transforms.ToTensor(),normalize])
        global_transforms2_list.extend([transforms.ToTensor(),normalize])
        local_transforms_list.extend([transforms.ToTensor(),normalize])

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
