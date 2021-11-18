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
# model = smp.Unet()

# model = smp.Unet('resnet34', encoder_weights='imagenet')


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]



# '''load the best model'''
# model = torch.load('./best_model.pth')    
model = torch.load('./best_model_transfer_t3.pth')    

'''presudo labeling the dataset'''
# thre = 0.8590604
# for i, data in enumerate(train_loader, 0):
#     length = len(train_loader)
#     inputs, name = data
#     label_name = name[0].split('\\')[-1]
#     mask_name = label_name+'gt.png'
    
#     inputs = inputs.to(device)
#     output = best_model.predict(inputs)
#     mask_pred = (output.squeeze().cpu().numpy())
#     mask_pred = (255*(mask_pred>thre)).astype(np.uint8)
#     label_mask= (inputs.squeeze().cpu().numpy()*255).astype(np.uint8)
    
#     cv2.imwrite(os.path.join('../results_joints/',label_name), mask_pred)
#     cv2.imwrite(os.path.join('../results_joints/',mask_name), label_mask)

'''fine tune the model with presudo labeling dataset'''
'''forzen the encoder?'''
for countp, parents in enumerate(model.children()):
    for count, child in enumerate(parents.children()):
        if countp>=1:
            print(child)
            # break
        else:       
            for param in child.parameters():
                param.requires_grad=False
        print("Child ",countp, count," is frozen now")
           
optimizer = torch.optim.Adam([ 
    dict(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.000005),
])


'''evaluate and save'''
'''evaluation test'''
data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'
valid_dataset = DefectDataset(data_pathes, istrain=False, debug=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

max_ap = 0
for epoch in range(10):
    losses = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
    
        # forward + backward
        outputs = model.forward(inputs)
        loss_out = loss(outputs, labels)
        loss_out.backward()
        optimizer.step()
        # 每训练1个batch打印一次loss和准确率
        losses.append( loss_out.item())
    print('loss is', np.mean(losses))

    

    labels_out, scores_out = [], []
    for i, data in enumerate(valid_loader, 0):
        length = len(valid_loader)
        inputs, labels = data
        inputs = inputs.to(device)
        output = model.predict(inputs)
        mask_pred = (output.squeeze().cpu().numpy())
        label_out = (labels.squeeze().cpu().numpy())
        labels_out.append(label_out)
        scores_out.append(mask_pred)
    
    labels_out = np.array(labels_out)
    scores_out = np.array(scores_out)    
    labels_out = (np.reshape(labels_out, [-1])).astype(np.int32)
    scores_out = np.reshape(scores_out, [-1])
    average_precision = average_precision_score(labels_out, scores_out)
    print('ap now is', average_precision)
    if average_precision > max_ap:       
        np.save('t2_t1_label_transfer1au_en.npy',labels_out)
        np.save('t2_t1_score_transfer1au_en.npy',scores_out)    
        max_ap = average_precision
        torch.save(model, './best_model_transfer_t3.pth')