"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : visualize.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Wed Apr 22 15:38:28 2020

==============================================================================
"""
import numpy as np
from matplotlib import pyplot as plt
import torch

from models.UNet import UNet
from utils.data import load_data
from utils.labels import labels

def get_color_image(img):
    '''
    Converts mask/label to RGB image
    Args: 'grayscale' encoded class masks
    Returns: RGB image
    ''' 
    id2color = {label.id : label.color for label in labels}
    img = img.cpu().numpy()
    out_img = np.array([[id2color[val] for val in row] for row in img], dtype='B')
    return out_img

def visualize(model_path, model, dataset='val', batch_size=1, shuffle=True):
    
    DATADIR = 'datasets/citys'
    
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
    
    file = torch.load(model_path)
    
    model = model.to(device)
    
    model.load_state_dict(file['model_state_dict'])
    print('Finished loading model!')
    model.eval()
    
    data_generator = load_data(DATADIR, batch_size=batch_size, shuffle=shuffle)
    val_generator = data_generator[dataset]
    
    data = next(iter(val_generator))
    imgs, mask = data[0].to(device), data[1].to(device)
    
    with torch.no_grad():    
        prediction = model(imgs)
    
    pred = torch.argmax(prediction, dim=1).cpu()
    
    mask = 255 * torch.squeeze(mask, dim=1)         # remove redundant channel
    imgs = imgs.permute(0,2,3,1).cpu()
    
    
    fig, ax = plt.subplots(nrows=batch_size, ncols=3)
    
    for j in range(batch_size):
    
        pred_img = get_color_image(pred[j])
        mask_img = get_color_image(mask[j])
        
        ax[j,0].imshow(imgs[j])
        ax[j,1].imshow(pred_img)
        ax[j,2].imshow(mask_img)
    
    np.vectorize(lambda ax:ax.axis('off'))(ax)      # disable axis
    
    cols = ['image', 'prediction', 'ground truth']  # titles
    
    for ax, col in zip(ax[0], cols):
        ax.set_title(col)                           # set titles
    
    plt.tight_layout()
    plt.show()
    
    return

if __name__ == '__main__':
    
    '''model'''
    model = UNet(n_classes=34,
                 depth=4,
                 wf=3,
                 batch_norm=True,
                 padding=True,
                 up_mode='upconv')
    
    visualize(model_path='weights/unet-id2.pt',
              model=model, dataset='val',
              batch_size=3,
              shuffle=False)