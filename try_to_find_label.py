# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:05:19 2021

@author: zhang
"""
import numpy as np
import os
import cv2
from collections import Counter   #引入Counter
import matplotlib.pyplot as plt
test = np.load('t2_t1_label_transfer1au_all.npy')

np.sum(test)

test_restore = (np.reshape(test, [68,224,224])).astype(np.float32)

neg_sum_list = []
neg_idxss = []
for idx,t in enumerate(test_restore):
    if np.sum(t):
        neg_sum_list.append(np.sum(t))
        neg_idxss.append(idx)
plt.imshow(test_restore[neg_idxss[0]]*255)

# 'rail_55_0.png'
# neg_paths = np.load('type1_negative_lists.npy')
# pos_paths = np.load('type1_positive_lists.npy')

# data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'

# neg_allsum_lists = []

# for i in range(len(neg_paths)):
#     # neg_paths[i] = 'rail_55_0.png'
#     path = os.path.join(data_pathes, 'crop_groundtruth', neg_paths[i])
#     label = cv2.imread(path,0)  
#     label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
#     label = 255*(label>200).astype(np.uint8)

#     label = label / 255.0
#     label = label.astype(np.float32)
#     print(np.sum(label))
#     neg_allsum_lists.append(np.sum(label))

# b = dict(Counter(neg_allsum_lists))
# print ([key for key,value in b.items()if value > 1])  #只展示重复元素
# print ({key:value for key,value in b.items()if value > 1})  #展现重复元素和重复次数

# neg_luck_idx = []

# for i in neg_sum_list:
#     idx = neg_allsum_lists.index(np.float(i))
#     neg_luck_idx.append(idx)

# head = []

# for i in range(len(neg_paths)):
#     if i not in neg_luck_idx:
#         head.append(i)
        
# all_neg_idxs = head + neg_luck_idx

# np.save('neg_idx_restore.npy', all_neg_idxs)


# test_label = np.load('ablation/test_label.npy')
# test_restore = (np.reshape(test, [68,224,224])).astype(np.float32)

# neg_sum_list = []
# neg_idxss = []
# for idx,t in enumerate(test_restore):
#     if np.sum(t):
#         neg_sum_list.append(np.sum(t))
#         neg_idxss.append(idx)
# plt.imshow(test_restore[neg_idxss[0]]*255)
