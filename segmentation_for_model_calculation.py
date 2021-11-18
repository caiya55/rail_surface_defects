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
import mmd_loss
from torchsummary import summary

''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

data_path = '../results_joints'
join_train = joint_train_dataset(data_path) #训练数据集

train_loader = torch.utils.data.DataLoader(join_train, batch_size=1, shuffle=True) 

ENCODER = 'resnet34'
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
    # decoder_attention_type='scse',
    encoder_depth=2,
    decoder_channels= [512,128],  
)
# model = smp.Unet()

# model = smp.Unet('resnet34', encoder_weights='imagenet')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

summary(model.cuda(), (1,224,224))