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

import matplotlib.pyplot as plt

from models.UNet import UNet
from utils.data import load_data, checkfile
from utils.count_classes import WEIGHTS

weights = torch.FloatTensor(list(WEIGHTS.values()))

def train(data_generator, optim, epochs, print_every, save_model=True, weighted_loss=False):
    
    eval_ = {'loss': [], 'train_acc': [], 'valid_acc': []}
    
    for epoch in range(epochs):
        print('Epoch {}/{}, lr: {}'.format(epoch, epochs - 1, scheduler.get_lr()[0]))
        print('-' * 64)
        
        for phase in ['train']:#, 'val']:
            if phase == 'train':
    #            optimizer = scheduler(optimizer, epoch)
                model.train(True)           # Set model to training mode
            else:
                model.train(False)          # Set model to evaluate mode
                
            for t, (X, y) in enumerate(data_generator[phase]):
                
                X = X.to(device)                # [N, 1, H, W]
                y = y[:,0,:,:].to(device)       # [N, H, W] with class indices (0, 1)
                y = (y * 255).long()                    # otherwise loss throws an error
                
                prediction = model(X)           # [N, 19, H, W]
                
                if weighted_loss:
                    loss = F.cross_entropy(prediction, y, weights)
                else:
                    loss = F.cross_entropy(prediction, y)
                    
                optim.zero_grad()               # zero the parameter (weight) gradients
                
                if phase == 'train':
                    loss.backward()
                    optim.step()
                
                if t % print_every == 0:
                    print('[{}/{} ({:.0f}%)]\t{} Loss: {}'.format(
                    t * len(X), len(data_generator[phase].dataset),
                    100. * t / len(data_generator[phase]), phase, loss.item()))
                    
                eval_['loss'].append(loss)
        scheduler.step()
                
    print('Training done.\n') 
    
    if save_model:
        PATH = checkfile()
        torch.save(model.state_dict(), PATH)
        print('saved model -> {}'.format(PATH))
        
    return eval_
    
if __name__ == '__main__':
    
    '''data'''    
    data_generator = load_data('datasets/citys', batch_size=4)
        
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
    
    weights = weights.to(device)
    
    '''model'''
    model = UNet(n_classes=34,
                 depth=4,
                 wf=3,
                 batch_norm=True,
                 padding=True,
                 up_mode='upsample').to(device)
    
    from torchsummary import summary
    summary(model, input_size=(3, 1024, 2048))
    
    '''training'''
    optim = torch.optim.Adam(model.parameters(),
                             lr=1e-2,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=5e-4)    
    
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optim, step_size=1, gamma=0.1)
    
    eval_= train(data_generator,
                 optim=optim,
                 epochs=2,
                 print_every=20,
                 save_model=True,
                 weighted_loss=True)
    
    plt.plot(eval_['loss'])
    plt.grid()
    plt.ylabel('Loss')
    plt.xlabel('Iterations')