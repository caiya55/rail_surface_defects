# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:00:50 2020

@author: zhang
"""
import torch
import numpy as np
import segmentation_models_pytorch as smp
from my_dataset import DefectDataset, joint_dataset, joint_train_dataset, DefectDataset2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import average_precision_score
from mmd_loss import mmd_rbf
''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

ENCODER = 'se_resnext50_32x4d'
# ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['defects']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=1,
    decoder_attention_type='scse',
#    encoder_depth=4,
)
model = torch.load('./best_model_transfer.pth')    
# model = torch.load('./best_model.pth')   

features = []
def hook(module, input, output):
    features.append(output.clone().detach())
    
model.eval()

'''show and save'''
scores_out = []

for j in range(5):
    '''load source and target dataset'''
    # data_path = '../results_joints'
    # tgt_dataset = joint_train_dataset(data_path) #训练数据集
    # tgt_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size = 5, shuffle=True,drop_last = True) 
    
    data_pathes = '../RSDDs dataset/Type-II RSDDs dataset'
    source_dataset = DefectDataset2(data_pathes, istrain=True)
    source_loader = DataLoader(source_dataset, batch_size = 5, shuffle=True, num_workers=0, drop_last = True)

    # data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'
    # source_dataset = DefectDataset(data_pathes, istrain=True, debug=False)
    # source_loader = DataLoader(source_dataset, batch_size = 3, shuffle=True, num_workers=0,drop_last = True)

    data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'
    tgt_dataset = DefectDataset(data_pathes, istrain=True, debug=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size = 5, shuffle=True, num_workers=0,drop_last = True)
    
    source_iter = iter(source_loader)
    target_iter = iter(tgt_loader)
        
    '''evaluate and save'''
    for i in range(min(len(tgt_loader),len(source_loader))):
        sdata = next(source_iter)
        tdata = next(target_iter)
        inputs_s, label_s = sdata
        inputs_t,_,_ = tdata
        inputs_s = inputs_s.to(device)
        inputs_t = inputs_t.to(device)
        
        handle = model.segmentation_head.register_forward_hook(hook)
        y = model(inputs_s)
        source_feature = features[-1]
        handle.remove()
        
        handle1 = model.segmentation_head.register_forward_hook(hook)
        y = model(inputs_t)
        tgt_feature = features[-1]
        handle1.remove()
    
        s_f = source_feature[:,0,:,:].view([source_feature.size(0),-1])
        t_f = tgt_feature[:,0,:,:].view([source_feature.size(0),-1])    
        
        mmd_dis = mmd_rbf(s_f, t_f)
        scores_out.append(mmd_dis.cpu().numpy())
        # feature_s = model.segmentation_head.register_forward_hook(hook)
        # feature_t = model.segmentation_head(inputs_t)
        
        # inputs = inputs.to(device)
        # mask_pred = (output.squeeze().cpu().numpy())
        # label_out = (labels.squeeze().cpu().numpy())
        # labels_out.append(label_out)
        # scores_out.append(mask_pred)


scores_out = np.array(scores_out)
print('mean mmd is', np.mean(scores_out))
