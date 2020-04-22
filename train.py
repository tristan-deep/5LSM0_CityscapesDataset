"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : train.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Tue Apr 21 21:21:42 2020

==============================================================================
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from models.UNet import UNet

def checkfile(PATH='weights/unet-test%s.pt'):
    i=1
    while os.path.exists(PATH % i): # check if file exists
        i += 1            
    return PATH % i

def load_data(DATADIR, batch_size=1):
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    
    params = {'mode': 'fine',
              'target_type': 'semantic',
              'transform': transform,
              'target_transform': transform}
    
    train_set = datasets.Cityscapes(DATADIR, split='train', **params)
    train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    
    val_set = datasets.Cityscapes(DATADIR, split='val', **params)
    val_generator = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    data_generator = {"train": train_generator, "val": val_generator}
    
    return data_generator


def train(data_generator, optim, epochs, print_every, save_model=True):

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 38)
        
        for phase in ['train', 'val']:
            if phase == 'train':
    #            optimizer = scheduler(optimizer, epoch)
                model.train(True)           # Set model to training mode
            else:
                model.train(False)          # Set model to evaluate mode
                
            for t, (X, y) in enumerate(data_generator[phase]):
                
                X = X.to(device)                # [N, 1, H, W]
                y = y[:,0,:,:].to(device)       # [N, H, W] with class indices (0, 1)
                y = y.long()                    # otherwise loss throws an error
                
                prediction = model(X)           # [N, 19, H, W]
                
                loss = F.cross_entropy(prediction, y)
        
                optim.zero_grad()               # zero the parameter (weight) gradients
                
                if phase == 'train':
                    loss.backward()
                    optim.step()
                
                if t % print_every == 0:
                    print('[{}/{} ({:.0f}%)]\t{} Loss: {:.6f}'.format(
                    t * len(X), len(data_generator[phase]),
                    100. * t / len(data_generator[phase]), phase, loss.item()))
    
    if save_model:
        PATH = checkfile()
        torch.save(model.state_dict(), PATH)
    
if __name__ == '__main__':
    
    '''data'''    
    data_generator = load_data('datasets/citys', batch_size=1)
        
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
    
    '''model'''
    model = UNet(n_classes=19,
                 depth=2,
                 wf=2,
                 padding=True,
                 up_mode='upsample').to(device)
    
    '''training'''
    optim = torch.optim.Adam(model.parameters())    
    
    train(data_generator,
          optim=optim,
          epochs=1,
          print_every=10,
          save_model=True)