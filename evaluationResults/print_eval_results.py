"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : print_eval_results.py
                    
    Author(s)     : Tristan Stevens and Nadine Nijssen
    Date          : Thu Apr 30 14:07:24 2020

==============================================================================
"""
import sys
import json

sys.path.append('..')
from utils.labels import labels
  
# Opening JSON file 
#f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id1-4e-CE.json',)
#f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id2-10e-WCE.json',)
#f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id3-10e-WCE-d5-MS.json',)
#f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id5-4e-WCE.json',)
#f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id6-15e-WCE-d4-MS.json',)
f = open('evaluationResults/resultPixelLevelSemanticLabeling_unet-id7-baseline.json',)

# returns JSON object as a dictionary 
data = json.load(f)


# name to label object
name2label = {label.name : label for label in labels}


classScores = data['classScores']
averageScoreClasses = data['averageScoreClasses']
# print class scores
print("{:<15} {:<10}".format('classes','IoU'))
print("--------------------------")
for v in classScores.items():
    classes, iou = v
    if name2label[classes].trainId != 255:
        print("{:<15}: {val:>5.3f}".format(classes, val=iou))
print("--------------------------")
print("Score Average  : {val:>5.3f}".format(val=averageScoreClasses))
print("--------------------------")
print("\n")


categoryScores = data['categoryScores']
averageScoreCategories = data['averageScoreCategories']
# print category scores
print("{:<15} {:<10}".format('categories','IoU'))
print("--------------------------")
for v in categoryScores.items():
    categories, iou = v
    print("{:<15}: {val:>5.3f}".format(categories, val=iou))
print("--------------------------")
print("Score Average  : {val:>5.3f}".format(val=averageScoreCategories))
print("--------------------------")
  
  
# Closing file 
f.close() 
