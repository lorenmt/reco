from network.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models

def create_model(num_classes,representation_dim=None,backbone="resnet101",architecture="deeplabv3p"):
    """
    To do : add the default case where representation_dim=None (case where there is no representation head)
    change default architecture to string
    """
    
    
    if backbone=="resnet101":
        backbone=models.resnet101(pretrained=True)
    if architecture=="deeplabv3p":
        architecture=DeepLabv3Plus
    return architecture(backbone,num_classes=num_classes,output_dim=representation_dim)
    
     