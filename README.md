# 5LSM0_CityscapesDataset
Model solves the pixel-semantic labeling task on the Cityscapes dataset. https://www.cityscapes-dataset.com/

## Dataset
Data: go to [DOWNLOADS](https://www.cityscapes-dataset.com/downloads/) and start by downloading:
* Training, validation and test ground truth: [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB)
* Training, validation and test images: [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) (11GB)
* save the cityscapes datasets in the `./datasets/citys` dir.

## fast - SCNN
To run the fast-SCNN model used as baseline, follow instructions found [here](https://github.com/Tramac/Fast-SCNN-pytorch).