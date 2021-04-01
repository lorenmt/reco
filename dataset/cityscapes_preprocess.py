import glob
import os
from collections import namedtuple
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from module_list import *

root = 'dataset'
im_root = 'dataset/cityscapes/images'
label_root = 'dataset/cityscapes/labels'

os.makedirs(im_root + '/train')
os.makedirs(im_root + '/val')
os.makedirs(label_root + '/train')
os.makedirs(label_root + '/val')


train_im_list = glob.glob(root + '/leftImg8bit_trainvaltest/leftImg8bit/train/*')
counter = 0
for city in train_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((1024, 512))
        im.save(im_root + '/train/{}.png'.format(counter))
        counter += 1
print('Training RGB images processing has completed.')


val_im_list = glob.glob(root + '/leftImg8bit_trainvaltest/leftImg8bit/val/*')
counter = 0
for city in val_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((1024, 512))
        im.save(im_root + '/val/{}.png'.format(counter))
        counter += 1
print('Validation RGB images processing has completed.')

counter = 0
train_label_list = glob.glob(root + '/gtFine_trainvaltest/gtFine/train/*')
for city in train_label_list:
    label_list = glob.glob(city + '/*_labelIds.png')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((1024, 512), resample=Image.NEAREST)
        im.save(label_root + '/train/{}.png'.format(counter))
        counter += 1
print('Training Label images processing has completed.')

counter = 0
val_label_list = glob.glob(root + '/gtFine_trainvaltest/gtFine/val/*')
for city in val_label_list:
    label_list = glob.glob(city + '/*_labelIds.png')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((1024, 512), resample=Image.NEAREST)
        im.save(label_root + '/val/{}.png'.format(counter))
        counter += 1
print('Validation Label images processing has completed.')


# generate partial data with three seeds
for seed in range(3):
    np.random.seed(seed)
    random.seed(seed)
    create_folder(label_root + '/train_p1_{}'.format(seed))
    create_folder(label_root + '/train_p5_{}'.format(seed))
    create_folder(label_root + '/train_p25_{}'.format(seed))
    label_list = glob.glob(label_root + '/train/*')
    perc = [0.25, 0.05, 0.01]
    for i in range(2975):
        im  = np.array(Image.open(label_root + '/train/{}.png'.format(i)))
        void_class = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        label_class = np.unique(im).tolist()
        valid_class = [c for c in label_class if c not in void_class]
        for p in perc:
            im_ = np.zeros_like(im)
            for l in valid_class:
                label_mask = np.zeros_like(im)
                label_mask_ = im == l
                label_idx = np.transpose(np.nonzero(label_mask_))
                sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False)
                label_idx_ = label_idx[sample_ind]
                target_num = int(p * label_mask_.sum())
                label_mask[label_idx_[0, 0], label_idx_[0, 1]] = 1
                label_mask_true = label_mask

                # continuously grow mask until reach expected percentage
                while label_mask_true.sum() < target_num:
                    label_mask = cv2.dilate(label_mask, kernel=np.ones([5, 5]))
                    label_mask_true = label_mask * label_mask_
                im_[label_mask_true.astype(bool)] = l
            im_ = Image.fromarray(im_)
            im_.save(label_root + '/train_p{:d}_{}/{}.png'.format(int(p * 100), seed, i))

    create_folder(label_root + '/train_p0_{}'.format(seed))
    label_list = glob.glob(label_root + '/train/*')
    for i in range(2975):
        im = np.array(Image.open(label_root + '/train/{}.png'.format(i)))
        void_class = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        label_class = np.unique(im).tolist()
        valid_class = [i for i in label_class if i not in void_class]
        im_ = np.zeros_like(im)
        for l in valid_class:
            label_idx = np.transpose(np.nonzero(im == l))
            sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False)
            label_idx_ = label_idx[sample_ind]
            im_[label_idx_[:, 0], label_idx_[:, 1]] = l
        im_ = Image.fromarray(im_)
        im_.save(label_root + '/train_p0_{}/{}.png'.format(seed, i))

    print('Partial Label images for seed {} has completed.'.format(seed))

print('All Done.')
