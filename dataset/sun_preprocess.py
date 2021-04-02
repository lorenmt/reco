import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from PIL import Image
from module_list import *
import re


def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

root = 'dataset/sun'

# load true test index
idx_list = []
with open(root + '/test_idx.txt') as f:
    for l in f:
        idx_list.append(int(l))

# rename training images
train_list = glob.glob(root + '/SUNRGBD-train_images/*')
train_list.sort(key=natural_key, reverse=True)

for f in train_list:
    im_idx = int(f[f.rfind('-') + 1: f.rfind('.')])
    os.rename(f, f[:-14] + 'img-{:06d}.jpg'.format(im_idx + 5050))
print('Training images re-naming has completed.')


# rename additional training images
test_list = glob.glob(root + '/SUNRGBD-test_images/*')
for f in test_list:
    im_idx = int(f[f.rfind('-') + 1: f.rfind('.')])
    if im_idx not in idx_list:
        os.rename(f, root + '/SUNRGBD-train_images/' + f[-14:])
print('Additional Training images re-naming has completed.')


# rescaling training images
train_list = glob.glob(root + '/SUNRGBD-train_images/*')
for f in train_list:
    im = Image.open(f)
    im = im.resize((512, 384))
    im.save(f)
print('Training images processing has completed.')

# rescaling training images
test_list = glob.glob(root + '/SUNRGBD-test_images/*')
for f in test_list:
    im = Image.open(f)
    im = im.resize((512, 384))
    im.save(f)
print('Testing images processing has completed.')


# rescaling label images
label_list = glob.glob(root + '/sunrgbd_train_test_labels/*')
for f in label_list:
    im = Image.open(f)
    im = im.resize((512, 384), resample=Image.NEAREST)
    im.save(f)
print('Label images processing has completed.')


