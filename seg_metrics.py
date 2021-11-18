import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from my_dataset import DefectDataset, joint_dataset, joint_train_dataset, DefectDataset2
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer.npy', 't2_t1_score_transfer_en.npy',
#                 't2_t1_score_transfer_all.npy', 't2_t1_score_transfer1_all.npy','t2_t1_score_transfer1au_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer.npy', 't2_t1_label_transfer_en.npy',
#                 't2_t1_label_transfer_all.npy','t2_t1_label_transfer1_all.npy','t2_t1_label_transfer1au_all.npy']

# scores_path = ['scores_out_unet_dic.npy', 'scores_out_unet_dic_aug.npy','scores_out_unet_dic_att.npy',
#                 'scores_out_unet_dic_att_tranf.npy']

# labels_path = ['labels_out_unet_dic.npy', 'labels_out_unet_dic_aug.npy','labels_out_unet_dic_att.npy',
#                 'labels_out_unet_dic_att_tranf.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer.npy', 't2_t1_score_transfer_en.npy',
#                 't2_t1_score_transfer_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer.npy', 't2_t1_label_transfer_en.npy',
#                 't2_t1_label_transfer_all.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer_all.npy', 't2_t1_score_transfer1_self_learning.npy',
#                't2_t1_score_transfer1_all.npy','t2_t1_score_transfer1au_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer_all.npy', 't2_t1_label_transfer1_self_learning.npy',
#                't2_t1_label_transfer1_all.npy','t2_t1_label_transfer1au_all.npy']

# scores_path = ['scores_out_unet_jac.npy','scores_out_unet_dic.npy', 'scores_out_unet_focl.npy']

# labels_path = ['labels_out_unet_jac.npy','labels_out_unet_dic.npy', 'labels_out_unet_focl.npy']


scores_path = ['t1_t2_score.npy', 't1_t2_score_transfer_en.npy',
                't1_t2_score_transfer_all.npy', 't1_t2_score_transfer1_all.npy','t1_t2_score_transfer1au_all.npy']
labels_path = ['t1_t2_label.npy',  't1_t2_label_transfer_en.npy',
                't1_t2_label_transfer_all.npy','t1_t2_label_transfer1_all.npy','t1_t2_label_transfer1au_all.npy']

# scores_path = ['scores_out_unet_jac.npy','scores_out_unet_dic.npy', 'scores_out_unet_focl.npy']

# labels_path = ['labels_out_unet_jac.npy','labels_out_unet_dic.npy', 'labels_out_unet_focl.npy']

path = '../results_restore'

idx = 0



for idx in range(0, len(scores_path)):
    score_out =  np.load(scores_path[idx])
    label_out =  np.load(labels_path[idx])
    
    auc_out = roc(label_out, score_out)
    precision, recall, thresholds = precision_recall_curve(label_out, score_out)
    average_precision = average_precision_score(label_out, score_out)
    print('AP is', average_precision)
    f1 = (precision * recall) / (precision + recall + 1e-6)
    diff = np.abs(precision-recall).tolist()
    # yuzhi_idx = np.argmin(diff)
    yuzhi_idx = np.argmax(f1)
    
    precision_val = precision[yuzhi_idx]
    recall_val = recall[yuzhi_idx]
    print(' recall star is', recall_val)
    print(' precision star is', precision_val)
    
    th = thresholds[yuzhi_idx]
    score_out = (score_out>th)*255
    
    F1 = f1_score(label_out, score_out/255)
    print('F1 is', F1)
    
    jacard_coefficent = jaccard_score(label_out, score_out/255)
    
    print('JACC is', jacard_coefficent)
    
    # score_out = score_out.astype(np.uint8)
    # score_out= np.reshape(score_out, [641,224,224])
    # label_out= np.reshape(label_out, [641,224,224])
    # label_out = (label_out*255).astype(np.uint8)
    
    # plt.imshow(score_out[0])
    # plt.imshow(label_out[0])
    
    # pp = 64
    # out_name = os.path.join(path,str(pp)+'gt.png')
    # scor_name = os.path.join(path,str(pp)+'_'+str(idx)+'.png')
    
    # cv2.imwrite(out_name, label_out[pp])
    # cv2.imwrite(scor_name, score_out[pp])
    
# '''evaluation test'''
# idx = 0
# data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'
# valid_dataset = DefectDataset(data_pathes, istrain=False, debug=True)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
# for i, data in enumerate(valid_loader, 0):
#     length = len(valid_loader)
#     inputs, labels, name = data
#     print(name, idx)
#     idx = idx + 1