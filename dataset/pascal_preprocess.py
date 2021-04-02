import glob
import os
from collections import namedtuple
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from module_list import *


im_root = 'dataset/pascal/JPEGImages'
label_root = 'dataset/pascal/SegmentationClassAug'


for seed in range(3):
    np.random.seed(seed)
    random.seed(seed)

    create_folder(label_root + '_p1_{}'.format(seed))
    create_folder(label_root + '_p5_{}'.format(seed))
    create_folder(label_root + '_p25_{}'.format(seed))

    label_list = glob.glob(label_root + '/*.png')
    perc = [0.25, 0.05, 0.01]

    for i in range(len(label_list)):
        im  = np.array(Image.open(label_list[i]))
        im_id = (label_list[i][label_list[i].rfind('/') + 1: label_list[i].rfind('.')])
        void_class = [255]
        label_class = np.unique(im).tolist()
        valid_class = [c for c in label_class if c not in void_class]
        for p in perc:
            im_ = np.ones_like(im) * 255
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
            im_.save(label_root + '_p{:d}_{}/{}.png'.format(int(p * 100), seed, im_id))

    create_folder(label_root + '_p0_{}'.format(seed))
    label_list = glob.glob(label_root + '/*.png')
    for i in range(len(label_list)):
        im  = np.array(Image.open(label_list[i]))
        im_id = (label_list[i][label_list[i].rfind('/') + 1: label_list[i].rfind('.')])
        void_class = [255]
        label_class = np.unique(im).tolist()
        valid_class = [i for i in label_class if i not in void_class]
        im_ = np.ones_like(im) * 255
        for l in valid_class:
            label_idx = np.transpose(np.nonzero(im == l))
            sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False)
            label_idx_ = label_idx[sample_ind]
            im_[label_idx_[:, 0], label_idx_[:, 1]] = l
        im_ = Image.fromarray(im_)
        im_.save(label_root + '_p0_{}/{}.png'.format(seed, im_id))

    print('Partial Label images for seed {} has completed.'.format(seed))

print('All Done.')
