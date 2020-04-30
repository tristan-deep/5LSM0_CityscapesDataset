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

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

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

def visualize(model_weights, model, dataset='val', batch_size=1, shuffle=True):
    
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

def plot_loss(p):
    train_loss = p['train_loss']
    val_loss = p['val_loss']
    
    train_loss = torch.stack(train_loss).cpu().detach().numpy()
    val_loss = torch.stack(val_loss).cpu().detach().numpy()
    val_loss = val_loss.reshape(int(len(val_loss)/p['epoch']), -1)
    val_loss = np.mean(val_loss, axis=0)                # take mean of each epoch
    
    smooth_train_loss = movingaverage(train_loss, 30)   # smoothen train loss
    epochs = np.linspace(len(train_loss)/p['epoch'],len(train_loss),p['epoch'])
    
    plt.figure()
    plt.plot(train_loss, color='C1', alpha=0.5)
    plt.plot(smooth_train_loss, color='C1')
    plt.plot(epochs, val_loss, color='C0', linestyle='dashed', marker='^')
    
    plt.grid()
    plt.ylabel('Loss')
    plt.xlabel('Iterations ({} epochs)'.format(p['epoch']))
    plt.title(p['loss_function'])
    plt.legend(['train loss','smooth train loss','validation loss'])

if __name__ == '__main__':
    
    '''model'''
    model = UNet(n_classes=34,
                 depth=5,
                 wf=3,
                 batch_norm=True,
                 padding=True,
                 up_mode='upconv')
    
    # change to the model you want to visualize
    # unet-id1-4e-CE
#    model_weights = file = torch.load('weights/unet-id1-4e-CE.pt')
    # unet-id2-10e-WCE
#    model_weights = file = torch.load('weights/unet-id2-10e-WCE.pt')
    # unet-id3-10e-WCE-d5-MS
    model_weights = file = torch.load('weights/unet-id3-10e-WCE-d5-MS.pt', map_location='cuda:0')
    # unet-id5-4e-WCE
#    model_weights = file = torch.load('weights/unet-id5-4e-WCE.pt')
    # unet-id6-15e-WCE-d4-MS
#    model_weights = file = torch.load('weights/unet-id6-15e-WCE-d4-MS.pt')
   
    plot_loss(model_weights)
    
    visualize(model_weights=model_weights,
              model=model, dataset='val',
              batch_size=3,
              shuffle=False)