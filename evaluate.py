"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : evaluate.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Wed Apr 22 15:38:28 2020

==============================================================================
"""
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.UNet import UNet
from train import load_data

PATH = 'weights/unet-test2.pt'
DATADIR = 'datasets/citys'

# this is not nice, try to find a way to extract colormappings from pytorch Cityscape dataloader
def colormap_cityscapes(n=20):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

'''device''' 
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print('using device:', device)

'''model'''
model = UNet(n_classes=19,
             depth=4,
             wf=3,
             padding=True,
             up_mode='upsample').to(device)

model.load_state_dict(torch.load(PATH))
model.eval()

data_generator = load_data(DATADIR, batch_size=1)
val_generator = data_generator['val']

cmap = colormap_cityscapes()

with torch.no_grad():
    for i, data in enumerate(val_generator):
        imgs, mask = data[0].to(device), data[1].to(device)
        prediction = model(imgs)
        pred = torch.argmax(prediction, dim=1).cpu()#.squeeze(0).cpu().data.numpy()
                
        pred_imgs = [cmap[p] for p in pred]
        for pred_img in pred_imgs:
            plt.imshow(pred_img)
            plt.show()
        
        break