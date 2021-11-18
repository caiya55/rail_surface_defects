import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import average_precision_score

font = {'family' : 'SimHei',
        'weight' : 'bold',
        'size'   : '12'}
 
''' calculate the auc value for lables and scores'''
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

'''load the five curves'''
# scores_path = ['scores_out_unet_ef_dic.npy', 'scores_out_pan_ef_dic.npy', 'scores_out_pspnet_ef_dic.npy',
#                'scores_out_deeplab_ef_dic.npy', 'scores_out_linknet_ef_dic.npy']
# labels_path = ['labels_out_unet_ef_dic.npy', 'labels_out_pan_ef_dic.npy', 'labels_out_pspnet_ef_dic.npy',
#                'labels_out_deeplab_ef_dic.npy', 'labels_out_linknet_ef_dic.npy']

# scores_path = ['scores_out_unet_dic.npy', 'scores_out_pan_dic.npy', 'scores_out_pspn_dic.npy',
#                'scores_out_deeplab_dic.npy', 'scores_out_linknet_dic.npy']
# labels_path = ['labels_out_unet_dic.npy', 'labels_out_pan_dic.npy', 'labels_out_pspn_dic.npy',
#                'labels_out_deeplab_dic.npy', 'labels_out_linknet_dic.npy']

# scores_path = ['scores_out_unet_jac.npy','scores_out_unet_dic.npy', 'scores_out_unet_focl.npy']

# labels_path = ['labels_out_unet_jac.npy','labels_out_unet_dic.npy', 'labels_out_unet_focl.npy']

# scores_path = ['scores_out_unet_dic.npy', 'scores_out_unet_dic_aug.npy','scores_out_unet_dic_att.npy',
#                 'scores_out_unet_dic_att_tranf.npy']

# labels_path = ['labels_out_unet_dic.npy', 'labels_out_unet_dic_aug.npy','labels_out_unet_dic_att.npy',
#                 'labels_out_unet_dic_att_tranf.npy']

# scores_path = ['t2_scores_out_unet_dic_noaug.npy','t2_scores_out_unet_dic.npy','t2_scores_out_unet_dic_att.npy',
#                 't2_scores_out_unet_dic_att_tranf.npy']
# labels_path = ['t2_labels_out_unet_dic_noaug.npy','t2_labels_out_unet_dic.npy','t2_labels_out_unet_dic_att.npy',
#                 't2_labels_out_unet_dic_att_tranf.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer.npy', 't2_t1_score_transfer_en.npy',
#                 't2_t1_score_transfer_all.npy', 't2_t1_score_transfer1_all.npy','t2_t1_score_transfer1au_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer.npy', 't2_t1_label_transfer_en.npy',
#                 't2_t1_label_transfer_all.npy','t2_t1_label_transfer1_all.npy','t2_t1_label_transfer1au_all.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer.npy', 't2_t1_score_transfer_en.npy',
#                 't2_t1_score_transfer_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer.npy', 't2_t1_label_transfer_en.npy',
#                 't2_t1_label_transfer_all.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer_all.npy', 't2_t1_score_transfer1_self_learning.npy',
#                't2_t1_score_transfer1_all.npy','t2_t1_score_transfer1au_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer_all.npy', 't2_t1_label_transfer1_self_learning.npy',
#                't2_t1_label_transfer1_all.npy','t2_t1_label_transfer1au_all.npy']

# scores_path = ['t2_t1_score.npy', 't2_t1_score_transfer.npy', 't2_t1_score_transfer_en.npy',
#                 't2_t1_score_transfer_all.npy','t2_t1_score_transfer1_self_learning.npy',
#                't2_t1_score_transfer1_all.npy','t2_t1_score_transfer1au_all.npy']
# labels_path = ['t2_t1_label.npy', 't2_t1_label_transfer.npy', 't2_t1_label_transfer_en.npy',
#                 't2_t1_label_transfer_all.npy','t2_t1_label_transfer1_self_learning.npy',
#                't2_t1_label_transfer1_all.npy','t2_t1_label_transfer1au_all.npy']

# scores_path = ['t1_t2_score.npy', 't1_t2_score_transfer.npy', 't1_t2_score_transfer_en.npy',
#                 't1_t2_score_transfer_all.npy', 't1_t2_score_transfer1_all.npy','t1_t2_score_transfer1au_all.npy']
# labels_path = ['t1_t2_label.npy', 't1_t2_label_transfer.npy', 't1_t2_label_transfer_en.npy',
#                 't1_t2_label_transfer_all.npy','t1_t2_label_transfer1_all.npy','t1_t2_label_transfer1au_all.npy']

scores_path = ['t1_t2_score.npy', 't1_t2_score_transfer_en.npy',
                't1_t2_score_transfer_all.npy', 't1_t2_score_transfer1_all.npy','t1_t2_score_transfer1au_all.npy']
labels_path = ['t1_t2_label.npy',  't1_t2_label_transfer_en.npy',
                't1_t2_label_transfer_all.npy','t1_t2_label_transfer1_all.npy','t1_t2_label_transfer1au_all.npy']

colors = ['slategray', 'darkorange', 'brown',  'darkslategray', 'indigo', 'limegreen', 'yellow', 'orchid', 'lightseagreen','b']
def calculate(score_path, label_path):
    scores_out =  np.load(score_path)
    labels_out =  np.load(label_path)
    
    # auc_out = roc(labels_out, scores_out)
    fpr, tpr, thresholds  = roc_curve(labels_out, scores_out)
    precision, recall, thresholds = precision_recall_curve(labels_out, scores_out)
    
    average_precision = average_precision_score(labels_out, scores_out)
    return precision, recall, average_precision
# right_index = (tpr + (1 - fpr) - 1).tolist()
# yuzhi = max(right_index)
# index = right_index.index(max(right_index))
# tpr_val = tpr[index]
# fpr_val = fpr[index]
# th = thresholds[index]
# ## 绘制roc曲线图
# y_predt = 1*(scores_out<th)


# conf_mat= confusion_matrix(labels_out, y_predt, labels=[0, 1])
# print('Confusion matrix:\n', conf_mat)

# labels = ['No defects', 'Defects']
# cm = np.array(conf_mat)
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.colorbar()
# tick_marks = np.arange(len(labels))
# plt.xticks(tick_marks, labels)
# plt.yticks(tick_marks, labels)

# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, cm[i, j],
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")

# plt.tight_layout()
# plt.xlabel('Predicted')
# plt.ylabel('Expected')
# plt.savefig('conf.png',dpi=400)
# plt.show()
'''show the curves'''
plt.subplots(figsize=(7,5.5))
plt.rc('font', **font)
for i in range(len(labels_path)):
    precision, recall, average_precision = calculate(scores_path[i], labels_path[i])
    plt.plot( recall, precision, color=colors[i],lw=2)
    print('average precision AP is', average_precision)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid('True')

'''unet res 0.752  eff 0.606'''
'''pan res 0.738  eff 0.666'''
'''pspnet res 0.677  eff 0.669'''
'''deeplab res 0.747  eff 0.677'''
'''linknet res 0.667  eff 0.640'''

# plt.legend(['Unet, AP=0.752', 'Unet with data aug, AP=0.886', 'Unet with data aug and attention, AP=0.895',
#             'Unet- aug and attention and transfer AP=0.893','Balance line'])
# plt.legend(['Unet, AP=0.755', 'Unet with data aug, AP=0.819', 'Unet with data aug and attention, AP=0.820',
#             'Unet transfer and hist matched AP=0.617','Balance line'])

# plt.legend(['No adaptation, AP=0.103', 'Last layer adaptation, AP=0.624','Decoder adaptation AP=0.701',
#             'Encoder+Decoder adaptation AP=0.749','with histogram matching AP=0.858', 'with data augmentation AP=0.861', 'Balance line'])

# plt.legend(['No adaptation, AP=0.103', 'Last layer adaptation, AP=0.624','Decoder adaptation AP=0.701',
#             'Encoder+Decoder adaptation AP=0.749','Balance line'])

# plt.legend(['No adaptation, AP=0.103', 'Last layer ad +self-learning, AP=0.624','Decoder ad +self-learning AP=0.701',
#             'En+Decoder ad +self-learning AP=0.749','+ SA AP=0.778','+ PHM AP=0.858', 
#             '+ data augmentation AP=0.861','Balance line'])

plt.legend(['No adaptation, AP=0.105', '+Self-learning AP=0.191',
            '+SA AP=0.21','+PHM AP=0.291','+ Data augmentation AP=0.295','Balance line'])

# plt.legend(['无适应 AP=0.103', '+自学习 AP=0.524',
#             '+因果模型 AP=0.679','+目标域自学习 AP=0.701','+ 注意力机制 AP=0.738','+ 直方图匹配 AP=0.758', 
#             '+ 数据增强 AP=0.791','Balance line'])

# plt.legend(['Unet-Jaccard loss, AP=0.752', 'Unet-Dice loss, AP=0.727', 'Unet-Focal loss, AP=0.481','Balance line'])

plt.savefig('ablation/t12t2.png', dpi=400)
plt.show()

