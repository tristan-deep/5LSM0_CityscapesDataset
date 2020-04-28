"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : evaluate.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Sun Apr 26 09:52:09 2020

==============================================================================
"""
import numpy as np
import torch

from models.UNet import UNet
from train import load_data

from utils.metrics import calculate_IoU, calculate_IoU_train_classes

def evaluate(model_path, model, dataset='val', batch_size=1):
    
    DATADIR = 'datasets/citys'
    
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
    
    model = model.to(device)
    
    model.load_state_dict(torch.load(model_path))
    print('Finished loading model!')
    model.eval()
    
    data_generator = load_data(DATADIR, batch_size=batch_size)
    val_generator = data_generator[dataset]
    
    data = next(iter(val_generator))
    imgs, mask = data[0].to(device), data[1].to(device)
    
    with torch.no_grad():    
        prediction = model(imgs)
    
    pred = torch.argmax(prediction, dim=1).cpu()
    
    IoU, IoU_dict, IoU_average = calculate_IoU(mask, pred, 34)
    print('IoU per class: ')
    for key, value in IoU_dict.items():
        print(key, ' : ', value)
    print('IoU average for 34 classes: ', IoU_average)
    IoU_19_average = calculate_IoU_train_classes(IoU)
    print('IoU average for 19 classes: ', IoU_19_average)
    
    return

if __name__ == '__main__':
    
    '''model'''
    model = UNet(n_classes=34,
                 depth=4,
                 wf=3,
                 batch_norm=True,
                 padding=True,
                 up_mode='upsample')
    
    evaluate(model_path='weights/unet-test13.pt',
              model=model, dataset='val',
              batch_size=3)