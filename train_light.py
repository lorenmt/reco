import argparse


from network.deeplabv3.deeplabv3 import *
from build_data_cracks import BuildDataLoaders_semi_supervised_crack_dataset,get_color_mask
from module_list import *



import pytorch_lightning as pl
from lightning_model import ema_segmentation_model

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Full labels')

    #parser.add_argument('--gpu', default=0, type=int)
    parser=pl.Trainer.add_argparse_args(parser)

    parser=ema_segmentation_model.add_model_specific_args(parser)
    parser.add_argument('--path_to_dataset', default="C:/Users/lavra/Documents/GitHub/painting-cracks/addoration_of_lamb", type=str)#TO DO : change default name
    
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    dict_args=vars(args)
    model=ema_segmentation_model(**dict_args)

    builder_dataloader = BuildDataLoaders_semi_supervised_crack_dataset(args.path_to_dataset, ["190_101","191_132","192_37","100_35","341_34","215_138","215_184"],["126_112"],num_segments=args.num_segments)

    train_l_loader, train_u_loader,validation_loader = builder_dataloader.build(args.seed)
    dataloader=[train_l_loader,train_u_loader]
    trainer=pl.Trainer.from_argparse_args(parser)
    
    trainer.fit(model,dataloader,validation_loader)
