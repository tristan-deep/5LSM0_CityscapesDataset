# 5LSM0_CityscapesDataset
Model solves the pixel-semantic labeling task on the Cityscapes dataset. https://www.cityscapes-dataset.com/

## Dataset
Data: go to [DOWNLOADS](https://www.cityscapes-dataset.com/downloads/) and start by downloading:
* Training, validation and test ground truth: [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB)
* Training, validation and test images: [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) (11GB)
* save the cityscapes datasets in the `./datasets/citys` dir.

## Models
All pretrained models are based on a simple U-net architecture and stored in [weights](weights) folder. The PyTorch implementation has been taken from this [repo](https://github.com/jvanvugt/pytorch-unet).
If you want to run predictions use [save_prediction.py](save_prediction.py) and make sure the designated folder for the predictions exists.

## Results
![](figures/predictions-id3-val3.png?raw=true)

