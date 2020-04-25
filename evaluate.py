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

PATH = 'weights/unet-test4.pt'
DATADIR = 'datasets/citys'
bs=3

# still need to find a way to extract colormappings from pytorch Cityscape dataloader

'''device''' 
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print('using device:', device)

'''model'''
model = UNet(n_classes=34,
             depth=4,
             wf=3,
             padding=True,
             up_mode='upsample').to(device)

model.load_state_dict(torch.load(PATH))
model.eval()

data_generator = load_data(DATADIR, batch_size=bs)
val_generator = data_generator['val']

with torch.no_grad():
    for i, data in enumerate(val_generator):
        imgs, mask = data[0].to(device), data[1].to(device)
        
        prediction = model(imgs)
        pred = torch.argmax(prediction, dim=1).cpu()#.squeeze(0).cpu().data.numpy()

        imgs = np.transpose(imgs.cpu(), [0, 2, 3, 1])        
        
        fig, ax = plt.subplots(nrows=bs, ncols=3)
        
        for j in range(bs):
            ax[j,0].imshow(imgs[j].cpu())
            ax[j,1].imshow(np.squeeze(pred[j]),vmin=0, vmax=34)
            ax[j,2].imshow(np.squeeze(mask[j].cpu()*255), vmin=0, vmax=34)
        
        np.vectorize(lambda ax:ax.axis('off'))(ax)      # disable axis
        
        cols = ['image', 'prediction', 'ground truth']  # titles
        
        for ax, col in zip(ax[0], cols):
            ax.set_title(col)                           # set titles
        
        plt.show()
        
        break