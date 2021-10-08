# ReCo - Regional Contrast

This repository contains the source code of ReCo and baselines from the paper, [Bootstrapping Semantic Segmentation with Regional Contrast](https://arxiv.org/abs/2104.04465), introduced by [Shikun Liu](https://shikun.io/), [Shuaifeng Zhi](https://shuaifengzhi.com/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/).

Check out [our project page](https://shikun.io/projects/regional-contrast) for more qualitative results. 


## Updates
**Aug. 2021** -- For PyTorch 1.9 users, you may encounter a small bug in `ColorJitter` data augmentation function due to version mismatch, which I have now provided a solution in the comment.

**Oct. 2021** -- Updated DeepLabv2 backbone.

## Datasets
ReCo was implemented by *PyTorch 1.7* and *TorchVision 0.8*, and evaluated with three datasets: **CityScapes**, **PASCAL VOC** and **SUN RGB-D** in the full label mode, among which **CityScapes** and **PASCAL VOC** are additionally evaluated in the partial label mode. 

- For CityScapes, please download the original dataset from the [official CityScapes site](https://www.cityscapes-dataset.com/downloads/): `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`. Create and extract them to the corresponding `dataset/cityscapes` folder.
- For Pascal VOC, please download the original training images from the [official PASCAL site](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar): `VOCtrainval_11-May-2012.tar` and the augmented labels [here](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip): `SegmentationClassAug.zip`. Extract the folder `JPEGImages` and `SegmentationClassAug` into the corresponding `dataset/pascal` folder.
- For SUN RGB-D, please download the train dataset [here](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz): `SUNRGBD-train_images.tgz`, test dataset [here](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz): `SUNRGBD-test_images.tgz` and labels [here](https://github.com/ankurhanda/sunrgbd-meta-data/raw/master/sunrgbd_train_test_labels.tar.gz): `sunrgbd_train_test_labels.tar.gz`. Extract and place them into the corresponding `dataset/sun` folder. 

After making sure all datasets having been downloaded and placed correctly, run each processing file `python dataset/{DATASET}_preprocess.py` to pre-process each dataset ready for the experiments. The preprocessing file also includes generating partial label for Cityscapes and Pascal dataset with three random seeds. Feel free to modify the partial label size and random seed to suit your own research setting.

For the lazy ones: just download the off-the-shelf pre-processed datasets here: [CityScapes](https://www.dropbox.com/sh/1eeq4qi9g2n6la2/AAD4IK1oskNPUzfTuusMqfb7a?dl=0), [Pascal VOC](https://www.dropbox.com/sh/gaoqumpylcci3he/AABjenlsGet060yhGXVxobE4a?dl=0) and [SUN RGB-D](https://www.dropbox.com/sh/miq8361xxbricp5/AAD8E74uWKwELbHmhAyGshCfa?dl=0).

## Training Supervised and Semi-supervised Models
In this paper, we introduce two novel training modes for semi-supervised learning.
1. Full Labels Partial Dataset: A sparse subset of training images has full ground-truth labels, with the remaining data unlabelled.
2. Partial Labels Full Dataset: All images have some labels, but covering only a sparse subset of pixels.

Running the following four scripts would train each mode with supervised or semi-supervised methods respectively:
```
python train_sup.py             # Supervised learning with full labels.
python train_semisup.py         # Semi-supervised learning with full labels.
python train_sup_partial.py     # Supervised learning with partial labels.
python train_semisup_patial.py  # Semi-supervised learning with partial labels.
```

### Important Flags
All supervised and semi-supervised methods can be trained with different flags (hyper-parameters) when running each training script. We briefly introduce some important flags for the experiments below.

| Flag Name        | Usage  |  Comments |
| ------------- |-------------| -----|
| `backbone`     | choose semantic segmentation backbone model: `deeplabv3p, deeplabv2`  |  DeepLabv3+ works better and faster  |
| `num_labels`     | number of labelled images in the training set, choose `0` for training all labelled images  | only available in the full label mode  |
| `partial`     |  percentage of labeled pixels for each class in the training set, choose `p0, p1, p5, p25` for training 1, 1%, 5%, 25% labelled pixel(s) respectively  | only available in the partial label mode |
| `num_negatives` | number of negative keys sampled for each class in each mini-batch | only applied when training with ReCo loss|
| `num_queries` | number of queries sampled for each class in each mini-batch | only applied when training with ReCo loss|
| `output_dim` | dimensionality for pixel-level representation | only applied when training with ReCo loss|
| `temp` | temperature used in contrastive learning | only applied when training with ReCo loss|
| `apply_aug` | semi-supervised methods with data augmentation, choose `cutout, cutmix, classmix` | only available in the semi-supervised methods; our implementations for [CutOut, CutMix](https://arxiv.org/abs/1906.01916) and [ClassMix](https://arxiv.org/abs/2007.07936)|
| `weak_threshold` | weak threshold `delta_w` in active sampling | only applied when training with ReCo loss|
| `strong_threshold` | strong threshold `delta_s` in active sampling | only applied when training with ReCo loss|
| `apply_reco` | toggle on or off | apply our proposed ReCo loss|

Training ReCo + ClassMix with the fewest **full** label setting in each dataset (the least appeared classes in each dataset have appeared in 5 training images):
```
python train_semisup.py --dataset pascal --num_labels 60 --apply_aug classmix --apply_reco
python train_semisup.py --dataset cityscapes --num_labels 20 --apply_aug classmix --apply_reco
python train_semisup.py --dataset sun --num_labels 50 --apply_aug classmix --apply_reco
```

Training ReCo + ClassMix with the fewest **partial** label setting in each dataset (each class in each training image only has 1 labelled pixel):
```
python train_semisup_partial.py --dataset pascal --partial p0 --apply_aug classmix --apply_reco
python train_semisup_partial.py --dataset cityscapes --partial p0 --apply_aug classmix --apply_reco
python train_semisup_partial.py --dataset sun --partial p0 --apply_aug classmix --apply_reco
```

Training ReCo + Supervised with all labelled data:
```
python train_sup.py --dataset {DATASET} --num_labels 0 --apply_reco
```

Training with ReCo is expected to require 12 - 16G of memory in a single GPU setting. All the other baselines can be trained under 12G in a single GPU setting.

## Visualisation on Pre-trained Models
We additionally provide the pre-trained baselines and our method for 20 labelled Cityscapes and 60 labelled Pascal VOC, as examples for visualisation. The precise mIoU performance for each model is listed in the following table. The pre-trained models will produce the exact same qualitative results presented in the original paper.  

 |  | Supervised        |  ClassMix  |  ReCo + ClassMix |
-------| ------------- |-------------| -----|
CityScapes (20 Labels) | 38.10 [[link]](https://www.dropbox.com/s/q6txvxnlhjzood0/cityscapes_label20_sup.pth?dl=0) | 45.13 [[link]](https://www.dropbox.com/s/eyrs1n9vifikfas/cityscapes_label20_semi_classmix.pth?dl=0) | 50.14 [[link]](https://www.dropbox.com/s/aa1lcsrxujo9t4v/cityscapes_label20_semi_classmix_reco.pth?dl=0) |
Pascal VOC (60 Labels) | 36.06 [[link]](https://www.dropbox.com/s/lhmlvea3kmqrfc7/pascal_label60_sup.pth?dl=0) | 53.71 [[link]](https://www.dropbox.com/s/v6nlbmg9apboc0c/pascal_label60_semi_classmix.pth?dl=0) | 57.12 [[link]](https://www.dropbox.com/s/xsxawpix5mtpi69/pascal_label60_semi_classmix_reco.pth?dl=0) |

Download the pre-trained models with the links above, then create and place them into the folder `model_weights` in this repository. Run `python visual.py` to visualise the results.

### Other Notices
1. We observe that the performance for the fully labelled semi-supervised CityScapes is not stable across different machines, for which all methods may drop 2-5% performance, though the ranking keeps the same. Different GPUs in the same machine do not affect the performance.  The performance for the other datasets in the full label mode, and the performance for all datasets in the partial label mode is consistent and stable.
2. The current data sampling strategy in full label mode is designed for choosing very few labelled data (< 20%), to achieve a balanced class distribution (sample new images which must contain the least sampled class in previous sampled images, and the number of classes in each image much be more than a defined threshold). If you plan to use the code to evaluate on more labelled data, please change the `unique_num` in each `get_DATASET_idx` function to 1 (number of class threshold in each image), or you can completely remove this sampling strategy by sampling training data randomly.
3.  Please use `--seed 0, 1, 2` to accurately reproduce/compare our results with the exactly same labelled and unlabelled split we used in our experiments.

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:
```
@article{liu2021reco,
    title={Bootstrapping Semantic Segmentation with Regional Contrast},
    author={Liu, Shikun and Zhi, Shuaifeng and Johns, Edward and Davison, Andrew J},
    journal={arXiv preprint arXiv:2104.04465},
    year={2021}
}
```

## Contact
If you have any questions, please contact sk.lorenmt@gmail.com.




