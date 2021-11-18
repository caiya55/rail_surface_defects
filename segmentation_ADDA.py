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
from ADDA_models import  Discriminator
from torch import nn
from thop import profile
from flopth import flopth
from ptflops import get_model_complexity_info

''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc


    
data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'
data_pathes_type1 = '../RSDDs dataset/Type-II RSDDs dataset'

source_dataset = DefectDataset(data_pathes, istrain=True)
target_dataset = DefectDataset2(data_pathes_type1, istrain=False, debug=False)

source_loader = DataLoader(source_dataset, batch_size=10, shuffle=True, num_workers=0,drop_last=True) 
target_loader = DataLoader(target_dataset, batch_size=10, shuffle=True, num_workers=0,drop_last=True)


ENCODER = 'resnet34'
# ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['defects']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''source model'''
model = smp.PAN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=1,
    # decoder_attention_type='scse',
    # encoder_depth=4,
    # decoder_channels= [512,128,64,32],  
)

source_model = torch.load('./best_model_t2_unet.pth')    

'''target model'''
model = smp.PAN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=1,
    # decoder_attention_type='scse',
    # encoder_depth=4,
    # decoder_channels= [512,128,64,32],  
)

target_model = torch.load('./best_model_t2_unet.pth')    

'''discriminator'''
discriminator = Discriminator()
discriminator = discriminator.to(DEVICE)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
# optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001),])
optimizer = torch.optim.Adam(
    target_model.encoder.parameters(),
    lr=3e-5, betas=(.5, .999), weight_decay=2.5e-5)
d_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=3e-5, betas=(.5, .999), weight_decay=2.5e-5)

criterion = nn.CrossEntropyLoss()
dm_loss = torch.nn.MSELoss()
source_model.eval()
target_model.encoder.train()
discriminator.train()
    
valid_epoch = smp.utils.train.ValidEpoch(
    target_model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)
max_score = 0
train_flag = 5
for i in range(0, 20):
    '''train epoch'''
    n_iters = min(len(source_loader), len(target_loader))
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    d_losses, g_losses = [], []
    for iter_i in range(n_iters):
        source_data, source_target = source_iter.next()
        target_data, target_target = target_iter.next()
        source_data = source_data.to(DEVICE)
        target_data = target_data.to(DEVICE)
        bs = source_data.size(0)
    
        D_input_source = source_model.encoder(source_data)
        D_input_target = target_model.encoder(target_data)
        D_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).to(DEVICE)
        D_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).to(DEVICE)
    
        # train Discriminator
        '''deep matching loss'''
        DMLoss = dm_loss(D_input_source[4], D_input_target[4]) + \
                    dm_loss(D_input_source[3],D_input_target[3]) + \
                    dm_loss(D_input_source[2],D_input_target[2])
        D_output_source = discriminator(D_input_source[-1])
        D_output_target = discriminator(D_input_target[-1])
        D_output = torch.cat([D_output_source, D_output_target], dim=0)
        D_target = torch.cat([D_target_source, D_target_target], dim=0)
        d_loss = criterion(D_output, D_target) + DMLoss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.append(d_loss.item())
    
        # train Target
        if train_flag > 2:
            train_flag = 0
            D_input_target = target_model.encoder(target_data)
            D_output_target = discriminator(D_input_target[-1])
            loss = criterion(D_output_target, D_target_source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            g_losses.append(loss.item())
        else:
            train_flag = train_flag + 1
    print('train g and d losses are', np.mean(d_losses), np.mean(g_losses))
    
    '''valid epoch'''
    print('\nEpoch: {}'.format(i))
    valid_logs = valid_epoch.run(target_loader)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(target_model, './best_t1_model_MD.pth')
        print('Model saved!')
    
    if i == 15:
        optimizer.param_groups[0]['lr'] = 4e-5
        print('Decrease decoder learning rate to 1e-5!')
        
# '''load the best model'''
best_model = torch.load('./best_t1_model_MD.pth')    
'''show and save'''
labels_out, scores_out = [], []
'''evaluate and save'''
for i, data in enumerate(target_loader, 0):
    length = len(target_loader)
    inputs, labels = data
    inputs = inputs.to(device)
    output = target_model.predict(inputs)
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
# np.save('t1_MD_label.npy',labels_out)
# np.save('t1_MD_score.npy',scores_out)

discriminator = Discriminator()
discriminator = discriminator.to(DEVICE)

inputs = torch.randn(1, 256, 14, 14)
flops, params = profile(discriminator)

sum_flops = flopth(discriminator, in_size=(256,14,14))

flops, params = get_model_complexity_info(discriminator, (256, 14, 14), as_strings=True, print_per_layer_stat=True)

print(sum_flops)