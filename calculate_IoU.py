# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:09:46 2020

@author: s166744
"""
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

def calculate_IoU(target, prediction, n_classes=34):
    # create confusion matrix with shape (n_classes, 2, 2)
    # (0,0) = true negatives, (1,1) = true positives
    # (0,1) = false positives, (1,0) = false negatives
    CF = multilabel_confusion_matrix(np.squeeze(target.cpu()*255).reshape(-1), 
                                     np.squeeze(prediction).reshape(-1),
                                     labels=list(range(n_classes)))
    
    # calculate the intersection over union for each class
    IoU = np.zeros(n_classes)
    for n in range(n_classes):
        intersection = CF[n,1,1]
        union = CF[n,1,1] + CF[n,0,1] + CF[n,1,0]
        if union == 0:
            IoU[n] = 0
        else:
            IoU[n] = intersection/union
    
    # average IoU    
    IoU_n_classes = np.average(IoU)
    print(IoU_n_classes)
    
    return IoU, IoU_n_classes
