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

for i, data in enumerate(val_generator):
    imgs, mask = data[0].to(device), data[1].to(device)
    prediction = model(imgs)
    
    break