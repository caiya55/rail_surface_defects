# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:00:50 2020

@author: zhang
"""
import torch
import numpy as np
import segmentation_models_pytorch as smp
from my_dataset import DefectDataset, DefectDataset2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import average_precision_score
from torchvision.models import resnet50
from thop import profile

''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

data_pathes = '../RSDDs dataset/Type-II RSDDs dataset'
# data_pathes_type1 = '../RSDDs dataset/Type-I RSDDs dataset'

train_dataset = DefectDataset2(data_pathes, istrain=True)
valid_dataset = DefectDataset2(data_pathes, istrain=False, debug=False)
# valid_dataset = DefectDataset(data_pathes_type1, istrain=False, debug=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0) 
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False, num_workers=0)


ENCODER = 'resnet34'
# ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['defects']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = None #'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=1,
    # decoder_attention_type='scse',
    # encoder_depth=4,
    # decoder_channels= [512,128,64,32],  
)
# model = smp.Unet()

# model = smp.Unet('resnet34', encoder_weights='imagenet')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

max_score = 0

# input = torch.randn(1, 1, 224, 224)
# flops, params = profile(model, inputs=(input, ))
# print('flops is',  flops//1000000)
# print('params is', params//1000000)
for i in range(0, 20):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model_t2_unet.pth')
        print('Model saved!')
        # labels_out, scores_out = [], []
        # '''evaluate and save'''
        # for i, data in enumerate(valid_loader, 0):
        #     length = len(valid_loader)
        #     inputs, labels = data
        #     inputs = inputs.to(device)
        #     output = model.predict(inputs)
        #     mask_pred = (output.squeeze().cpu().numpy())
        #     label_out = (labels.squeeze().cpu().numpy())
        #     labels_out.append(label_out)
        #     scores_out.append(mask_pred)
        
        # labels_out = np.array(labels_out)
        # scores_out = np.array(scores_out)    

        


    if i == 15:
        optimizer.param_groups[0]['lr'] = 4e-5
        print('Decrease decoder learning rate to 1e-5!')
        
# '''load the best model'''
best_model = torch.load('./best_model_t2.pth')    
'''show and save'''
labels_out, scores_out = [], []
'''evaluate and save'''
for i, data in enumerate(valid_loader, 0):
    length = len(valid_loader)
    inputs, labels = data
    inputs = inputs.to(device)
    output = best_model.predict(inputs)
    mask_pred = (output.squeeze().cpu().numpy())
    label_out = (labels.squeeze().cpu().numpy())
    labels_out.append(label_out)
    scores_out.append(mask_pred)

labels_out = np.array(labels_out)
scores_out = np.array(scores_out)    
''''reshape the array'''
labels_out_re  = labels_out[0]
for i in range(1,len(labels_out)):
    if len(labels_out[i].shape) == 3:
        labels_out_re = np.concatenate([labels_out_re,labels_out[i]],axis=0)
    else:
        tmp = labels_out[i]
        tmp = tmp[None, ...]
        labels_out_re = np.concatenate([labels_out_re,tmp],axis=0)
labels_out = (np.reshape(labels_out_re, [-1])).astype(np.int32)

scores_out_re  = scores_out[0]
for i in range(1,len(scores_out)):
    if len(scores_out[i].shape) == 3:
        scores_out_re = np.concatenate([scores_out_re,scores_out[i]],axis=0)
    else:
        tmp = scores_out[i]
        tmp = tmp[None, ...]
        scores_out_re = np.concatenate([scores_out_re,tmp],axis=0)
scores_out = np.reshape(scores_out_re, [-1])

auc_out = roc(labels_out, scores_out)
fpr, tpr, _ = roc_curve(labels_out, scores_out)
average_precision = average_precision_score(labels_out, scores_out)
print('precision is', average_precision)
# accuracy = accuracy_score(labels_out, scores_out)
plt.plot(fpr,tpr)
plt.show()    
np.save('t2_pan_label.npy',labels_out)
np.save('t2_pan_score.npy',scores_out)





