
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
import os



from data_augmentation import data_augmentation_labeled,data_augmentation_unlabeled,preprocess_ImageNet_normalization
import torch.utils.data.sampler as sampler
from module_list import *



#data_augmentation=importlib.reload(data_augmentation)

def pil_loader(path: str) -> Image.Image:
    # from pytorch https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cracks_label_map(mask):
    """
    transform the input image as a label
    """
    mask_flatten=mask.reshape(-1,mask.shape[2])

    mask_map = np.zeros(shape=(mask_flatten.shape[0]),dtype=np.uint8)

    mask_map[np.all(mask_flatten==[255,255,255],axis=1)]=1
    mask_map[np.all(mask_flatten==[255,0,0],axis=1)]=255
    return mask_map.reshape(mask.shape[0],mask.shape[1])

# --------------------------------------------------------------------------------
# Create dataset in PyTorch format
# --------------------------------------------------------------------------------

class DatasetUnlabeled(Dataset):
    def __init__(self, root, file_names, transformation=data_augmentation_unlabeled,
        crop_size=(160, 160), scale_size=(0.5, 1.0), partial_seed=None):
        self.root = os.path.expanduser(root)
        self.transformation= transformation
        self.crop_size = crop_size
        self.file_names = file_names
        self.scale_size = scale_size
        self.partial_seed = partial_seed

    def __getitem__(self, index):
        
        image_root=pil_loader(f"{self.root}/images/{self.file_names[index]}.jpg")
        
        image = self.transformation(image_root,self.crop_size) 
        
        return image#delete the unused channel dimention for the label

    def __len__(self):
        return len(self.file_names)


class DatasetLabeled(DatasetUnlabeled):
    
    def __init__(self,*args, transformation=data_augmentation_labeled,**kwargs):
        DatasetUnlabeled.__init__(self, *args,transformation=transformation,**kwargs)

    def __getitem__(self, index):

        image_root=pil_loader(f"{self.root}/images/{self.file_names[index]}.jpg")
        label_root=pil_loader(f"{self.root}/label/{self.file_names[index]}.png")
       
        label_root = Image.fromarray(cracks_label_map(np.array(label_root)))
        
        
        if self.transformation:
            image, label = self.transformation(image_root, label_root, 
        crop_size=self.crop_size, scale_size=self.scale_size)
        else:
            image,label,_=preprocess_ImageNet_normalization(image_root,label_root,None)
        return image, label.squeeze(0)#delete the unused channel dimention for the label 



def get_unlabeled_names(data_path,image_names_labeled):
    
    image_file_labeled=[f"{name}.jpg" for name in image_names_labeled]#add the extension for the label name
    extract_image_name=lambda image_file:image_file.split(".")[0]# not sure it's the clean way to do it
    return [extract_image_name(image_file) for image_file in os.listdir(f"{data_path}/images/") if image_file not in image_file_labeled]#dont fit inside the screen

# --------------------------------------------------------------------------------
# Create data loader in PyTorch format
# --------------------------------------------------------------------------------

class BuildDataLoaders_semi_supervised_crack_dataset:
    def __init__(self, data_path, image_names_labeled,image_names_labeled_validation,num_segments=2):
        """
        TODO : arguments as input : should use args & put all the relevent informations (for exemple :  crop size, scale_size)
        TODO : found auto the labeled images
        """
        
        if data_path[-1]=="/":#delete the / at the end if there is any
            data_path=data_path[:-1]
        self.data_path = data_path
        self.im_size = [256, 256]
        self.crop_size = [160,160]
        self.num_segments = num_segments#num of defined classes
        self.scale_size = (0.5, 2.)
        self.batch_size = 4
        self.image_names_labeled=image_names_labeled
        self.image_names_labeled_validation=image_names_labeled_validation
        self.image_names_unlabeled=get_unlabeled_names(data_path,image_names_labeled+image_names_labeled_validation)
             #TO DO : test phase dataset self.test_idx = get_cracks_idx(self.data_path, train=False)


    def build(self, partial_seed=None):
        
        train_l_dataset = DatasetLabeled(self.data_path,self.image_names_labeled,
                                       crop_size=self.crop_size, scale_size=self.scale_size, partial_seed=partial_seed)
        train_u_dataset = DatasetUnlabeled(self.data_path, self.image_names_unlabeled,
                                       crop_size=self.crop_size, scale_size=(1.0, 1.0),partial_seed=partial_seed)
        test_l_dataset = DatasetLabeled(self.data_path,self.image_names_labeled_validation,
                                       transformation=None, partial_seed=partial_seed)
       
        torch.manual_seed(partial_seed)
        #TODO : replace 200 with its correspounding value
        num_epochs=200
        num_samples = self.batch_size * num_epochs  

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
        )

       
        train_u_loader = torch.utils.data.DataLoader(
            train_u_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_u_dataset,
                                            replacement=True,
                                            num_samples=num_samples),
            drop_last=True,
        )

        validation_l_loader = torch.utils.data.DataLoader(
            test_l_dataset,
            batch_size=5,
            drop_last=False,
        )

    
        return train_l_loader, train_u_loader,validation_l_loader




def create_cracks_label_colormap():
  """Creates a label colormap for visualisation of the results
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0]
  colormap[1] = [255, 255, 255]
  colormap[255]=[255,0,0]

  
  return colormap


def apply_color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1],mask.shape[2], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)

def get_color_mask(batch_mask):

    colormap=create_cracks_label_colormap()
    return apply_color_map(batch_mask,colormap)