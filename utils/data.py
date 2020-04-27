import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def checkfile(PATH='weights/unet-test%s.pt'):
    i=1
    while os.path.exists(PATH % i): # check if file exists
        i += 1            
    return PATH % i

def load_data(DATADIR, batch_size=1, shuffle=False):
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])
    
    params = {'mode': 'fine',
              'target_type': 'semantic',
              'transform': transform,
              'target_transform': transform}
    
    train_set = datasets.Cityscapes(DATADIR, split='train', **params)
    train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    
    val_set = datasets.Cityscapes(DATADIR, split='val', **params)
    val_generator = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)
    
    data_generator = {"train": train_generator, "val": val_generator}
    
    return data_generator