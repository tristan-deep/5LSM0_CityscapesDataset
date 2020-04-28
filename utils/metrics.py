"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : metrics.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Sat Apr 25 18:09:46 2020

==============================================================================
"""
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from utils.labels import labels

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
            IoU[n] = None
        else:
            IoU[n] = intersection/union
    
    # average IoU   
    IoU_no_nan = IoU[~np.isnan(IoU)]
    IoU_average = np.average(IoU_no_nan)
    
    IoU_dict = {class_id: IoU[class_id] for class_id in range(n_classes)}
    
    return IoU, IoU_dict, IoU_average


def calculate_IoU_train_classes(IoU, n_classes=34):
    id2trainId = {label.id : label.trainId for label in labels}
    
    IoU_19 = np.zeros(19)
    for i in range(n_classes):	
        i_trainID = id2trainId[i]
        if i_trainID != 255:
            IoU_19[i_trainID] = IoU[i]
    
    IoU_19_no_nan = IoU_19[~np.isnan(IoU_19)]
    IoU_19_average = np.average(IoU_19_no_nan)
    
    return IoU_19_average
