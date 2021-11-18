# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:49:02 2020

@author: zhang
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt 
'''find the difference between the two images'''

'''data path is ../RSDDs dataset/Type-II RSDDs dataset'''
data_pathes = '../RSDDs dataset/Type-II RSDDs dataset'

neg_paths = np.load('type2_negative_lists.npy')
neg_idxs = np.load('t2_neg_idxs.npy')

train_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs]
label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs]

dises = []
for i in range(len(label_neg)):
    label_name = label_neg[i]
    label = cv2.imread(label_name,0)      
    label = 1*(label>200).astype(np.uint8)
        
    label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
    # label = 255*(label>0.5)
    pos_sum = label.sum()
    dises.append(pos_sum)
          
num_bins = 50
# the histogram of the data  
n_type2, bins_type2, patches = plt.hist(dises, num_bins, facecolor='blue', alpha=0.5)  
# add a 'best fit' line  


'''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'

neg_paths = np.load('type1_negative_lists.npy')
neg_idxs = np.load('neg_idxs.npy')

train_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs]
label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs]

dises = []
for i in range(len(label_neg)):
    label_name = label_neg[i]
    label = cv2.imread(label_name,0)      
    label = 1*(label>200).astype(np.uint8)
        
    label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
    # label = 255*(label>0.5)
    pos_sum = label.sum()
    dises.append(pos_sum)
          
num_bins = 50
# the histogram of the data  
n_type1, bins_type1, patches = plt.hist(dises, num_bins, facecolor='blue', alpha=0.5)  
# add a 'best fit' line  


'''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
data_path =  '../results_joints'
data_paths = os.listdir(data_path)
paths = [ os.path.join(data_path, pa) for pa in data_paths if pa.endswith('pnggt.png')]

dises = []
neg_idxs = []
for i in range(len(paths)):
    label_name = paths[i]
    label_name = label_name[:-10]+'.png'
    label = cv2.imread(label_name,0)      
    label = 1*(label>200).astype(np.uint8)
        
    label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
    # label = 255*(label>0.5)
    pos_sum = label.sum()
    if pos_sum > 0:
        dises.append(pos_sum)
        neg_idxs.append(i)
          
num_bins = 50
# the histogram of the data  
n_type3, bins_type3, patches = plt.hist(dises, num_bins, facecolor='blue', alpha=0.5)  
# add a 'best fit' line  

n_type3 = n_type3/n_type3.sum()
n_type2 = n_type2/n_type2.sum()
n_type1 = n_type1/n_type1.sum()

plt.plot(bins_type3[1:], n_type3, color='green') 
plt.plot(bins_type2[1:], n_type2, color='navy')  
plt.plot(bins_type1[1:], n_type1, color='darkorange') 


plt.xlabel('Anormaly area percentage')  
plt.ylabel('Frequency')  
plt.title(r'Histogram of anromaly percentage')  
plt.grid(True)
plt.show()  

np.save('joint_neg.npy',neg_idxs)
