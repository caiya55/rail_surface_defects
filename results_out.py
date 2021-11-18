# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:00:50 2020

@author: zhang
"""

import torch
import numpy as np
import segmentation_models_pytorch as smp
from my_dataset import DefectDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc
