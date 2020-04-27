"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : Cityscapesloader_test.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Mon Apr 20 12:56:37 2020

==============================================================================
"""
import numpy as np
from matplotlib import pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, transforms

bs = 3
DATADIR = 'datasets/citys'

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

target_type = 'color'
dataset = datasets.Cityscapes(DATADIR, split='train', mode='fine', target_type=target_type, transform=transform, target_transform=transform)

trainloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

for i, data in enumerate(trainloader):
    imgs, mask = data[0], data[1]
    imgs = np.transpose(imgs, [0, 2, 3, 1])
    
    if target_type=='semantic':
        mask = np.squeeze(mask)
    elif target_type=='color':
        mask = np.transpose(mask, [0, 2, 3, 1])[:,:,:,:3]
        
    f, axarr = plt.subplots(bs, 2)
    for j in range(bs):
        axarr[j,0].imshow(imgs[j])
        axarr[j,1].imshow(mask[j])
        axarr[j,0].set_title('image')
        axarr[j,1].set_title('ground truth')
        axarr[j,0].axis('off')
        axarr[j,1].axis('off')
           
    plt.show()
    break
