import pytorch_lightning as pl
import torch.nn.functional as nn_Func
import torch.optim as optim
import torch

from build_data_cracks import get_color_mask
import copy

import choose_model
from data_augmentation import \
    data_augmentation_unsupervised_on_tensor
from module_list import PolyLR,compute_unsupervised_loss,compute_supervised_loss,compute_reco_loss,label_onehot\
    
from data_augmentation import \
    image_net_denormalisation

class ema_segmentation_model(pl.LightningModule):
    def __init__(self,**args):
        super().__init__()  
        self.save_hyperparameters("num_segments","lr","weight_decay","apply_reco","apply_aug","weak_threshold",
        "strong_threshold","num_negatives","num_queries","temp","output_dim","optimizer","max_epochs")
        print("model parameters : ")
        print(self.hparams)
        self.args=args


        backbone="resnet101" 
        architecture="deeplabv3p"

        
        self.model=choose_model.create_model(self.hparams["num_segments"],representation_dim=self.hparams["output_dim"],backbone=backbone,architecture=architecture)
        self.teacher_model = copy.deepcopy(self.model)#EMA(self.model, 0.99)
        self.alpha=0.99
        self.step_ema=0


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ema_segmentation_model")
        parser.add_argument('--num_segments', default=2, type=int)
        parser.add_argument('--lr', default=2.5e-3, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--reco_loss', dest='apply_reco', action='store_true')
        parser.add_argument('--no_reco_loss', dest='apply_reco', action='store_false')
        parser.set_defaults(max_epochs=50)
        
        parser.set_defaults(apply_reco=True)
        parser.add_argument('--optimizer', default='SGD', type=str)
        
        parser.add_argument('--apply_aug', default='cutmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
        parser.add_argument('--weak_threshold', default=0.7, type=float)
        parser.add_argument('--strong_threshold', default=0.97, type=float)
        parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
        parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
        parser.add_argument('--temp', default=0.5, type=float)
        parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
        #parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p')
        

        return parent_parser



    def configure_optimizers(self):
        
        if self.hparams["optimizer"]=="SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"], momentum=0.9, nesterov=True)
        
        elif self.hparams["optimizer"]=="AdamW":
            optimizer=optim.AdamW(self.model.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

        scheduler = PolyLR(optimizer, self.hparams["max_epochs"], power=0.9)

        return [optimizer],[scheduler]

    def forward(self, x):
        return self.teacher_model(x)

    def add_images_to_trainer(self,image_data,prediction,name,label_data=None):
        
        logits=torch.softmax(prediction,dim=1)
        _, pred_labels=torch.max(logits, dim=1)


        logits=logits[:,0,:,:].unsqueeze(dim=1).expand(-1,3,-1,-1)
        pred_labels=get_color_mask(pred_labels)


        if not(label_data is None):
            label_color=get_color_mask(label_data)
            self.logger.experiment.add_images(f"{name} label",label_color,global_step=self.current_epoch,dataformats="NHWC")
        
        self.logger.experiment.add_images(f"{name} images",image_net_denormalisation(image_data),global_step=self.current_epoch)
        self.logger.experiment.add_images(f"{name} logits",logits,global_step=self.current_epoch)
        self.logger.experiment.add_images(f"{name} prediction",pred_labels,global_step=self.current_epoch,dataformats="NHWC")


    def validation_step(self,batch,batch_idx):
        """
        TODO : add writer functions

        """
        test_l_data, test_l_label = batch
        self.teacher_model.eval()
        
        pred_l = self.teacher_model(test_l_data)
        
        logits=torch.softmax(pred_l.cpu(),dim=1)
        _, pred_labels_l=torch.max(logits, dim=1)
        
      
        sup_loss = compute_supervised_loss(pred_l, test_l_label)
        
        

        self.log("supervised_loss_validation", sup_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        self.add_images_to_trainer(test_l_data.cpu(),pred_l.cpu(),"test",label_data=test_l_label.cpu())
        
        
        return sup_loss
    


    def training_step(self, batch, batch_idx):
        train_l_dataset,train_u_data=batch
        train_l_data, train_l_label = train_l_dataset

      
        crop_size=(160,160)
        scale_size=(0.8,1.)
        
        with torch.no_grad():
            #create the dataset teaching the student network
            pred_u, _ = self.teacher_model(train_u_data)
            # prediction of the network is divided by 4 
            #pred_large shape : (Batch_size,n_classes,H,W)
            train_u_aug_logits, train_u_aug_label = torch.max(torch.softmax(pred_u, dim=1), dim=1)
            train_u_data, train_u_aug_label, train_u_aug_logits=data_augmentation_unsupervised_on_tensor(train_u_data, train_u_aug_label, train_u_aug_logits,
                crop_size, scale_size,mode=self.hparams["apply_aug"])
            

        # generate labelled and unlabelled predictions
        pred_l, rep_l = self.model(train_l_data)
        pred_u, rep_u = self.model(train_u_data)
        # for contrastive loss
        rep_all = torch.cat((rep_l, rep_u))
        
       
        # supervised-learning loss
        sup_loss = compute_supervised_loss(pred_l, train_l_label)
        
        # unsupervised-learning loss
        unsup_loss = compute_unsupervised_loss(pred_u, train_u_aug_label, train_u_aug_logits, self.hparams["strong_threshold"])

        # apply regional contrastive loss
        if self.hparams["apply_reco"]:
            with torch.no_grad():
                train_u_aug_mask = train_u_aug_logits.ge(self.hparams["weak_threshold"]).float()
                mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                mask_all = nn_Func.interpolate(mask_all, size=rep_all.shape[2:], mode='nearest')
                
                #convert the label to one-hot (can we use sparse encoding? )
                label_l = nn_Func.interpolate(label_onehot(train_l_label, self.hparams["num_segments"]), size=rep_all.shape[2:], mode='nearest')
                label_u = nn_Func.interpolate(label_onehot(train_u_aug_label, self.hparams["num_segments"]), size=rep_all.shape[2:], mode='nearest')
                label_all = torch.cat((label_l, label_u))

                prob_l = torch.softmax(pred_l, dim=1)
                prob_u = torch.softmax(pred_u, dim=1)
                prob_all = torch.cat((prob_l, prob_u))
                prob_all = nn_Func.interpolate(prob_all, size=rep_all.shape[2:], mode='nearest')
                #TODO : see if I can reduce the number of call to .interpolate
                
            reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, self.hparams["strong_threshold"],
                                          self.hparams["temp"], self.hparams["num_queries"], self.hparams["num_negatives"])
        else:
            reco_loss = torch.tensor(0.0)


        loss = sup_loss + unsup_loss + reco_loss


        self.log("total loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("supervised_loss", sup_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("unsupervised_loss", unsup_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("reco_loss", reco_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if batch_idx==1:

            self.add_images_to_trainer(train_l_data.cpu(),pred_l.cpu(),"train_l",label_data=train_l_label.cpu())
            self.add_images_to_trainer(train_u_data.cpu(),pred_u.cpu(),"train_u")
        
        
        
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        decay = min(1 - 1 / (self.step_ema + 1), self.alpha)#in the begining, the teacher model is very bad -> in update, more importance for student model

        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
            teacher_param.data = decay * teacher_param.data + (1 - decay) * student_param.data
        




