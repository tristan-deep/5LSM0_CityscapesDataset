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
    class_amount = {entry: [] for entry in range(n_classes)} 
    
    for i, (_, y) in enumerate(data_generator):
        y = (255 * y).int().to(device)    
        classes, counts = torch.unique(y, return_counts=True)
        
        for j in range(len(classes)):
            class_amount[classes[j].item()] =+ counts[j].item()
        
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
labels_dict={0: 12485, 1: 6115050, 2: 5631448, 3: 2372550, 4: 1628773, 5: 485216, 6: 1755448, 7: 51402716, 8: 11105883, 9: 389448, 10: 1127135, 
11: 32944412, 12: 1404422, 13: 931716, 14: 647, 15: 1168, 16: 716904, 17: 1527438, 18: 496, 19: 249084, 20: 621830, 21: 25239449, 22: 1365658, 23: 5223362, 24: 1690760, 25: 111354, 26: 8587667, 27: 166024, 28: 268578, 29: 338064, 30: 54641, 
31: 953315, 32: 347596, 33: 329214}
 
# using mu=0.15
WEIGHTS={
 0: 7.592658424500733,
 1: 1.3986781235297439,
 2: 1.4810644665616552,
 3: 2.345465743154862,
 4: 2.7216040986344847,
 5: 3.932592195613941,
 6: 2.646706974253243,
 7: 1.0,
 8: 1.0,
 9: 4.152455996327125,
 10: 3.0897520543213357,
 11: 1.0,
 12: 2.8698052390081963,
 13: 3.2801583010138082,
 14: 10.552595332713706,
 15: 9.961893463826435,
 16: 3.5422444078104673,
 17: 2.785839247208519,
 18: 10.818365700489679,
 19: 4.599396159259686,
 20: 3.6845196047468995,
 21: 1.0,
 22: 2.8977947054807522,
 23: 1.5562898133715715,
 24: 2.6842529372599415,
 25: 5.404472032408247,
 26: 1.0591039650205016,
 27: 5.00505399201311,
 28: 4.52404497367802,
 29: 4.293951121561306,
 30: 6.116401831526866,
 31: 3.2572409640408253,
 32: 4.266145462387306,
 33: 4.3204783529948125}
        
if __name__ == '__main__':
    labels_dict = count_classes()
    weights = create_class_weight(labels_dict)
    
#total = sum(labels_dict.values(), 0.0)
#a = {k: v / total for k, v in labels_dict.items()}