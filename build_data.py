from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import glob

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from module_list import *


# --------------------------------------------------------------------------------
# Define data augmentation
# --------------------------------------------------------------------------------
def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            #  color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  For PyTorch 1.9/TorchVision 0.10 users
            color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


# --------------------------------------------------------------------------------
# Define segmentation label re-mapping
# --------------------------------------------------------------------------------
def cityscapes_class_map(mask):
    # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = 255
    mask_map[np.isin(mask, [7])] = 0
    mask_map[np.isin(mask, [8])] = 1
    mask_map[np.isin(mask, [11])] = 2
    mask_map[np.isin(mask, [12])] = 3
    mask_map[np.isin(mask, [13])] = 4
    mask_map[np.isin(mask, [17])] = 5
    mask_map[np.isin(mask, [19])] = 6
    mask_map[np.isin(mask, [20])] = 7
    mask_map[np.isin(mask, [21])] = 8
    mask_map[np.isin(mask, [22])] = 9
    mask_map[np.isin(mask, [23])] = 10
    mask_map[np.isin(mask, [24])] = 11
    mask_map[np.isin(mask, [25])] = 12
    mask_map[np.isin(mask, [26])] = 13
    mask_map[np.isin(mask, [27])] = 14
    mask_map[np.isin(mask, [28])] = 15
    mask_map[np.isin(mask, [31])] = 16
    mask_map[np.isin(mask, [32])] = 17
    mask_map[np.isin(mask, [33])] = 18
    return mask_map


def sun_class_map(mask):
    # -1 is equivalent to 255 in uint8
    return mask - 1


# --------------------------------------------------------------------------------
# Define indices for labelled, unlabelled training images, and test images
# --------------------------------------------------------------------------------
def get_pascal_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_name = root + '/train_aug.txt'
    else:
        file_name = root + '/val.txt'
    with open(file_name) as f:
        idx_list = f.read().splitlines()

    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(21)
        label_fill = np.arange(21)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []
            mask = np.array(Image.open(root + '/SegmentationClassAug/{}.png'.format(idx)))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            unique_num = len(mask_unique)   # number of unique classes

            # sample image if it includes the lowest appeared class and with more than 3 distinctive classes
            if len(labeled_idx) == 0 and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 3:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowest appearance
            label_fill = np.where(label_counter == label_counter.min())[0]

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list


def get_cityscapes_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/images/train/*.png')
    else:
        file_list = glob.glob(root + '/images/val/*.png')
    idx_list = [int(file[file.rfind('/') + 1: file.rfind('.')]) for file in file_list]

    if train:
        labeled_idx = []
        save_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)
        label_counter = np.zeros(19)
        label_fill = np.arange(19)
        while len(labeled_idx) < label_num:
            if len(idx_list_) > 0:
                idx = idx_list_.pop()
            else:
                idx_list_ = save_idx.copy()
                idx = idx_list_.pop()
                save_idx = []

            mask = cityscapes_class_map(np.array(Image.open(root + '/labels/train/{}.png'.format(idx))))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            unique_num = len(mask_unique)  # number of unique classes

            # sample image if it includes the lowest appeared class and with more than 12 distinctive classes
            if len(labeled_idx) == 0 and unique_num >= 12:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            elif np.any(np.in1d(label_fill, mask_unique)) and unique_num >= 12:
                labeled_idx.append(idx)
                label_counter[mask_unique] += 1
            else:
                save_idx.append(idx)

            # record any segmentation index with lowest occurrence
            label_fill = np.where(label_counter == label_counter.min())[0]
        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list


def get_sun_idx(root, train=True, label_num=5):
    root = os.path.expanduser(root)
    if train:
        file_list = glob.glob(root + '/SUNRGBD-train_images/*.jpg')
        idx_list = [int(file[file.rfind('-') + 1: file.rfind('.')]) for file in file_list]
    else:
        file_list = glob.glob(root + '/SUNRGBD-test_images/*.jpg')
        idx_list = [int(file[file.rfind('-') + 1: file.rfind('.')]) for file in file_list]

    if train:
        labeled_idx = []
        idx_list_ = idx_list.copy()
        random.shuffle(idx_list_)

        # create a label dictionary class_list [[Image_ID for class 0], [Image_ID for class 1], ...]
        class_list = [[] for _ in range(37)]
        for i in range(len(idx_list_)):
            idx = idx_list_[i]
            mask = sun_class_map(np.array(Image.open(root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(idx))))
            mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
            for k in mask_unique:
                class_list[k].append(idx)

        label_counter = np.zeros(37)
        label_fill = np.arange(37)
        ignore_val = []
        ignore_mask = np.ones(37, dtype=bool)  # store any semantic id has sampled all possible images
        while len(labeled_idx) < label_num:
            if len(class_list[label_fill[0]]) > 0:
                idx = class_list[label_fill[0]].pop()
            else:
                ignore_val.append(label_fill[0])
                ignore_mask[ignore_val] = False

            # sample image by the current lowest appeared class
            if idx not in labeled_idx:
                labeled_idx.append(idx)
                mask = sun_class_map(np.array(Image.open(root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(idx))))
                mask_unique = np.unique(mask)[:-1] if 255 in mask else np.unique(mask)  # remove void class
                label_counter[mask_unique] += 1

                # record any segmentation index with lowest occurrence
                label_fill = np.where(label_counter == label_counter[ignore_mask].min())[0]

            ignore_ind = [np.where(label_fill == i) for i in ignore_val if i in label_fill]
            if len(ignore_ind) > 0:  # ignore index when reaching all available images
                label_fill = np.delete(label_fill, ignore_ind)

        return labeled_idx, [idx for idx in idx_list if idx not in labeled_idx]
    else:
        return idx_list


# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.dataset = dataset
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed

    def __getitem__(self, index):
        if self.dataset == 'pascal':
            image_root = Image.open(self.root + '/JPEGImages/{}.jpg'.format(self.idx_list[index]))
            if self.apply_partial is None:
                label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[index]))
            else:
                label_root = Image.open(self.root + '/SegmentationClassAug_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index],))

            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

        if self.dataset == 'cityscapes':
            if self.train:
                image_root = Image.open(self.root + '/images/train/{}.png'.format(self.idx_list[index]))
                if self.apply_partial is None:
                    label_root = Image.open(self.root + '/labels/train/{}.png'.format(self.idx_list[index]))
                else:
                    label_root = Image.open(self.root + '/labels/train_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index]))
                label_root = Image.fromarray(cityscapes_class_map(np.array(label_root)))
            else:
                image_root = Image.open(self.root + '/images/val/{}.png'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/labels/val/{}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(cityscapes_class_map(np.array(label_root)))
            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

        if self.dataset == 'sun':
            if self.train:
                image_root = Image.open(self.root + '/SUNRGBD-train_images/img-{:06d}.jpg'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(sun_class_map(np.array(label_root)))
            else:
                image_root = Image.open(self.root + '/SUNRGBD-test_images/img-{:06d}.jpg'.format(self.idx_list[index]))
                label_root = Image.open(self.root + '/sunrgbd_train_test_labels/img-{:06d}.png'.format(self.idx_list[index]))
                label_root = Image.fromarray(sun_class_map(np.array(label_root)))
            image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
            return image, label.squeeze(0)

    def __len__(self):
        return len(self.idx_list)


# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------
class BuildDataLoader:
    def __init__(self, dataset, num_labels):
        self.dataset = dataset
        if dataset == 'pascal':
            self.data_path = 'dataset/pascal'
            self.im_size = [513, 513]
            self.crop_size = [321, 321]
            self.num_segments = 21
            self.scale_size = (0.5, 1.5)
            self.batch_size = 10
            self.train_l_idx, self.train_u_idx = get_pascal_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_pascal_idx(self.data_path, train=False)

        if dataset == 'cityscapes':
            self.data_path = 'dataset/cityscapes'
            self.im_size = [512, 1024]
            self.crop_size = [512, 512]
            self.num_segments = 19
            self.scale_size = (1.0, 1.0)
            self.batch_size = 2
            self.train_l_idx, self.train_u_idx = get_cityscapes_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_cityscapes_idx(self.data_path, train=False)

        if dataset == 'sun':
            self.data_path = 'dataset/sun'
            self.im_size = [385, 513]
            self.crop_size = [321, 321]
            self.num_segments = 37
            self.scale_size = (0.5, 1.5)
            self.batch_size = 5
            self.train_l_idx, self.train_u_idx = get_sun_idx(self.data_path, train=True, label_num=num_labels)
            self.test_idx = get_sun_idx(self.data_path, train=False)

        if num_labels == 0:  # using all data
            self.train_l_idx = self.train_u_idx

    def build(self, supervised=False, partial=None, partial_seed=None):
        train_l_dataset = BuildDataset(self.data_path, self.dataset, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)
        train_u_dataset = BuildDataset(self.data_path, self.dataset, self.train_u_idx,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=True, apply_partial=partial, partial_seed=partial_seed)
        test_dataset    = BuildDataset(self.data_path, self.dataset, self.test_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200  # for total 40k iterations with 200 epochs

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
        )

        if not supervised:
            train_u_loader = torch.utils.data.DataLoader(
                train_u_dataset,
                batch_size=self.batch_size,
                sampler=sampler.RandomSampler(data_source=train_u_dataset,
                                              replacement=True,
                                              num_samples=num_samples),
                drop_last=True,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
        )
        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, train_u_loader, test_loader


# --------------------------------------------------------------------------------
# Create Color-mapping for visualisation
# --------------------------------------------------------------------------------
def create_cityscapes_label_colormap():
  """Creates a label colormap used in CityScapes segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def create_pascal_label_colormap():
  """Creates a label colormap used in Pascal segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = 255 * np.ones((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0]
  colormap[1] = [128, 0, 0]
  colormap[2] = [0, 128, 0]
  colormap[3] = [128, 128, 0]
  colormap[4] = [0, 0, 128]
  colormap[5] = [128, 0, 128]
  colormap[6] = [0, 128, 128]
  colormap[7] = [128, 128, 128]
  colormap[8] = [64, 0, 0]
  colormap[9] = [192, 0, 0]
  colormap[10] = [64, 128, 0]
  colormap[11] = [192, 128, 0]
  colormap[12] = [64, 0, 128]
  colormap[13] = [192, 0, 128]
  colormap[14] = [64, 128, 128]
  colormap[15] = [192, 128, 128]
  colormap[16] = [0, 64, 0]
  colormap[17] = [128, 64, 0]
  colormap[18] = [0, 192, 0]
  colormap[19] = [128, 192, 0]
  colormap[20] = [0, 64, 128]
  return colormap


def create_sun_label_colormap():
  """Creates a label colormap used in SUN RGB-D segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [148, 65, 137]
  colormap[1] = [255, 116, 69]
  colormap[2] = [86, 156, 137]
  colormap[3] = [202, 179, 158]
  colormap[4] = [155, 99, 235]
  colormap[5] = [161, 107, 108]
  colormap[6] = [133, 160, 103]
  colormap[7] = [76, 152, 126]
  colormap[8] = [84, 62, 35]
  colormap[9] = [44, 80, 130]
  colormap[10] = [31, 184, 157]
  colormap[11] = [101, 144, 77]
  colormap[12] = [23, 197, 62]
  colormap[13] = [141, 168, 145]
  colormap[14] = [142, 151, 136]
  colormap[15] = [115, 201, 77]
  colormap[16] = [100, 216, 255]
  colormap[17] = [57, 156, 36]
  colormap[18] = [88, 108, 129]
  colormap[19] = [105, 129, 112]
  colormap[20] = [42, 137, 126]
  colormap[21] = [155, 108, 249]
  colormap[22] = [166, 148, 143]
  colormap[23] = [81, 91, 87]
  colormap[24] = [100, 124, 51]
  colormap[25] = [73, 131, 121]
  colormap[26] = [157, 210, 220]
  colormap[27] = [134, 181, 60]
  colormap[28] = [221, 223, 147]
  colormap[29] = [123, 108, 131]
  colormap[30] = [161, 66, 179]
  colormap[31] = [163, 221, 160]
  colormap[32] = [31, 146, 98]
  colormap[33] = [99, 121, 30]
  colormap[34] = [49, 89, 240]
  colormap[35] = [116, 108, 9]
  colormap[36] = [161, 176, 169]
  return colormap


def create_nyuv2_label_colormap():
  """Creates a label colormap used in NYUv2 segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 255]
  colormap[1] = [233,  89,  48]
  colormap[2] = [0, 218,  0]
  colormap[3] = [149, 0, 240]
  colormap[4] = [222, 241,  24]
  colormap[5] = [255, 206, 206]
  colormap[6] = [0, 224., 229]
  colormap[7] = [10., 136., 204]
  colormap[8] = [117,  29,  41]
  colormap[9] = [240,  35, 235]
  colormap[10] = [0, 167, 156.]
  colormap[11] = [249, 139,   0]
  colormap[12] = [225, 229, 194]
  return colormap


def color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)

