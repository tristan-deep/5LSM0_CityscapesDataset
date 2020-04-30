"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : save_prediction.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Thu Apr 30 10:45:11 2020

==============================================================================
"""
import numpy as np
from matplotlib import pyplot as plt
import torch

from models.UNet import UNet
from utils.data import load_data

import cv2
import os, sys

sys.path.append('cityscapesScripts-master')
from cityscapesscripts.helpers.csHelpers import *
sys.path.append('..')


def prediction(groundTruthImgList, model_weights, model, dataset='val', batch_size=1, shuffle=True):
    
    DATADIR = 'datasets/citys'
    
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
       
    model = model.to(device)
    
    model.load_state_dict(model_weights['model_state_dict'])
    print('Finished loading model!')
    model.eval()
    
    data_generator = load_data(DATADIR, batch_size=batch_size, shuffle=shuffle)
    val_generator = data_generator[dataset]
    
    for i, (X, y) in enumerate(val_generator):
        imgs = X.to(device)
        
        with torch.no_grad():    
            prediction = model(imgs)
        
        pred = torch.argmax(prediction, dim=1).cpu()
        	
        # convert to right format to save prediciton image
        image_to_save = torch.squeeze(pred, dim=0).numpy()
        
        # get name of prediction image to save
        csFile = getCsFileInfo(groundTruthImgList[i])
        # save the prediction images in the 'results' folder
        filePattern = "results/val/unet-id6-15e-WCE-d4-MS/{}_{}_{}_pred.png".format(csFile.city, csFile.sequenceNb, csFile.frameNb)
        # change to the results set you want to save predictions for
            # unet-id1-4e-CE
            # unet-id2-10e-WCE
            # unet-id3-10e-WCE-d5-MS
            # unet-id5-4e-WCE
            # unet-id6-15e-WCE-d4-MS
        
        
        # save prediction image
        cv2.imwrite(filePattern, image_to_save)
        
#        if i == 4:
#            break
    print('Prediction images saved.')
    
    return


def get_gt_file_names():
    # ground truth file names
#    cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'datasets','citys')
    groundTruthSearch  = os.path.join(cityscapesPath , "gtFine" , "val" , "*", "*_gtFine_labelIds.png" )
    
    groundTruthImgList = glob.glob(groundTruthSearch)
    if not groundTruthImgList:
        printError("Cannot find any ground truth images to use for evaluation. Searched for: {}".format(groundTruthSearch))
    
    return groundTruthImgList


if __name__ == '__main__':
    
    '''model'''
    model = UNet(n_classes=34,
                 depth=4,
                 wf=3,
                 batch_norm=True,
                 padding=True,
                 up_mode='upconv')
    
    # change to the model you want to save predictions for
    # unet-id1-4e-CE
#    model_weights = file = torch.load('weights/unet-id1-4e-CE.pt')
    # unet-id2-10e-WCE
#    model_weights = file = torch.load('weights/unet-id2-10e-WCE.pt')
    # unet-id3-10e-WCE-d5-MS
#    model_weights = file = torch.load('weights/unet-id3-10e-WCE-d5-MS.pt', map_location='cuda:0')
    # unet-id5-4e-WCE
#    model_weights = file = torch.load('weights/unet-id5-4e-WCE.pt')
    # unet-id6-15e-WCE-d4-MS
    model_weights = file = torch.load('weights/unet-id6-15e-WCE-d4-MS.pt')
    
    groundTruthImgList = get_gt_file_names()
    
    prediction(groundTruthImgList,
              model_weights=model_weights,
              model=model, dataset='val',
              batch_size=1,
              shuffle=False)
    