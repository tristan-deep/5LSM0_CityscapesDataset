"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : count_classes.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Sun Apr 26 16:03:00 2020

==============================================================================
"""

import torch
import math

from utils.data import load_data

def count_classes(data_set='train', batch_size=100):
    '''device''' 
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('using device:', device)
    

    data_generator = load_data('datasets/citys', batch_size=batch_size)[data_set]
    
    n_classes = 34
    class_amount = {entry: 0 for entry in range(n_classes)} 
    
    for i, (_, y) in enumerate(data_generator):
        y = (255 * y).int().to(device)    
        classes, counts = torch.unique(y, return_counts=True)
        
        for j in range(len(classes)):
            class_amount[classes[j].item()] += counts[j].item()
        
    print(class_amount)
    
    return class_amount

def create_class_weight(labels_dict,mu=0.15):
    '''
    labels_dict : {ind_label: count_label}
    mu : parameter to tune 
    '''
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        
    print(class_weight)
    return class_weight

    
''' RESULTS '''
labels_dict={0: 704950, 1: 286002726, 2: 81359604, 3: 94111150, 4: 83752079, 
             5: 17818704, 6: 75629728, 7: 2036416525, 8: 336090793, 
             9: 39065130, 10: 11239214, 11: 1260636120, 12: 36199498, 
             13: 48454166, 14: 547202, 15: 17860177, 16: 3362825, 
             17: 67789506, 18: 499872, 19: 11477088, 20: 30448193, 
             21: 879783988, 22: 63949536, 23: 221979646, 24: 67326424, 
             25: 7463162, 26: 386328286, 27: 14772328, 28: 12990290, 
             29: 2493375, 30: 1300575, 31: 12863955, 32: 5449152, 
             33: 22861233}
 
# using mu=0.15
WEIGHTS={0: 7.191087967559193, 
         1: 1.1854582246454664, 
         2: 2.442580682416604, 
         3: 2.2969830363495363, 
         4: 2.413598572610761, 
         5: 3.9612108747632466, 
         6: 2.515610133483388, 
         7: 1.0, 
         8: 1.024078226165426, 
         9: 3.176229313658234, 
         10: 4.422050653693638, 
         11: 1.0, 
         12: 3.252414315556169, 
         13: 2.960841246696081, 
         14: 7.444396824598142, 
         15: 3.9588860810748474, 
         16: 5.628678172356099, 
         17: 2.6250521626826067, 
         18: 7.534862780221965, 
         19: 4.401106866710359, 
         20: 3.425432917678947, 
         21: 1.0, 
         22: 2.683365294558388, 
         23: 1.4388738739041291, 
         24: 2.6319067772818796, 
         25: 4.831480381886469, 
         26: 1.0, 
         27: 4.148703865974567, 
         28: 4.277257411589833, 
         29: 5.927822352479835, 
         30: 6.578653092517841, 
         31: 4.287030352592263, 
         32: 5.145999566649913, 
         33: 3.712016972968436}
        
if __name__ == '__main__':
    labels_dict = count_classes()
    weights = create_class_weight(labels_dict)
    
#total = sum(labels_dict.values(), 0.0)
#a = {k: v / total for k, v in labels_dict.items()}