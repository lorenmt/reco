
from re import I
import torch
import random
from PIL import Image


import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from module_list import generate_unsup_data


def image_net_denormalisation(x):
    x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    return x


def tensor_to_pil(im, label, logits):
    im = image_net_denormalisation(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits


def random_rescale(image,label,logits,scale_size):
    """
    rescale the image & associated label& logits 
    works on cpu & should work on gpu
    
    """
    if isinstance(image,Image.Image):
        raw_w, raw_h=image.size
    else:#if pytorch tensor
        raw_w, raw_h=image.size()[-2:]
  

    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    if label is not None:
        label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    return image,label,logits

def random_crop(image,label,logits,crop_size,fill_value=255):
    
    #TODO : move the originial_size parameter to the parameters
    # Add padding if the image size is less than crop size
    # fill_value correpound to an invalid value
    
    if isinstance(image,Image.Image):
        resized_size=image.size
    else:#if pytorch tensor
        resized_size=image.size()[-2:]

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        
        if label is not None:
            label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=fill_value, padding_mode='constant')#fill with invalid values
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    
    if label is not None:
        label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    return image,label,logits


def random_flipping(image,label,logits):
    if torch.rand(1) > 0.5:
        image = transforms_f.hflip(image)
        if label is not None:
            label = transforms_f.hflip(label)
        if logits is not None:
            logits = transforms_f.hflip(logits)

    return image,label,logits


def apply_basic_augmentation(image,label,logits):
    # Random color jitter
    if torch.rand(1) > 0.2:
        color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  #For PyTorch 1.9/TorchVision 0.10 users
        #color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25)) #if above is not working
        image = color_transform(image)

    # Random Gaussian filter
    if torch.rand(1) > 0.5:
        #sigma = random.uniform(0.15, 1.15)
        image=transforms.GaussianBlur((1,1),sigma=(0.15, 1.15))(image)#vs image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image,label,logits



def preprocess_ImageNet_normalization(image,label,logits):

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    if label is not None:
        label = (transforms_f.to_tensor(label) * 255).long()
        label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image,label,logits
    
    



def transform(image, label,logits, crop_size=(160, 160), scale_size=(0.8, 1.0), augmentation=True):
    """
    Separate the data augmentation part & preprocessing part 
    TO DO : 
    - see if I can adapt the functions to don't pass None arguments (use *args ??)
    - see if we can put the data augmentation part away from the dataset building part 
    """
    
    image,label,logits=random_rescale(image,label,logits,scale_size)
    if crop_size!=-1:
        image,label,logits=random_crop(image,label,logits,crop_size)
   
    # random augmentation
    if augmentation:
        image,label,_=apply_basic_augmentation(image,label,None)
        image,label,logits=random_flipping(image,label,logits)

    
    image,label,logits=preprocess_ImageNet_normalization(image,label,logits)
    
    if logits is not None:
        return image, label, logits
    if label is not None:
        return image,label
    else:
        return image
    

def data_augmentation_labeled(image,label,crop_size=(160, 160), scale_size=(0.8, 1.0)):
    image,label,_=random_rescale(image,label,None,scale_size)
    image,label,_=random_crop(image,label,None,crop_size)
    image,label,_=apply_basic_augmentation(image,label,None)
    image,label,_=random_flipping(image,label,None)
    image,label,_=preprocess_ImageNet_normalization(image,label,None)
    return image,label


def data_augmentation_unlabeled(image,crop_size=(160, 160)):

    image,_,_=random_crop(image,None,None,crop_size)
    image,_,_=preprocess_ImageNet_normalization(image,None,None)

    return image

def data_augmentation_unsupervised_on_tensor(train_u_data, pseudo_labels, pseudo_logits,crop_size, scale_size,mode):
    
    # random scale images first 
    train_u_data,pseudo_labels, pseudo_logits=random_rescale(train_u_data, pseudo_labels, pseudo_logits,scale_size)
    if crop_size!=-1:
        train_u_data, pseudo_labels, pseudo_logits=random_crop(train_u_data, pseudo_labels, pseudo_logits,crop_size,fill_value=-1.)
    
    # apply mixing strategy: cutout, cutmix or classmix
    train_u_data, pseudo_labels, pseudo_logits = \
     generate_unsup_data(train_u_data, pseudo_labels, pseudo_logits, mode=mode)


    return train_u_data, pseudo_labels, pseudo_logits