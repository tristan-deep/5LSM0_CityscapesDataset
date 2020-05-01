# 5LSM0_CityscapesDataset
Model solves the pixel-semantic labeling task on the Cityscapes dataset. https://www.cityscapes-dataset.com/

## Dataset
Data: go to [DOWNLOADS](https://www.cityscapes-dataset.com/downloads/) and start by downloading:
* Training, validation and test ground truth: [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) (241MB)
* Training, validation and test images: [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) (11GB)
* save the cityscapes datasets in the `./datasets/citys` dir.

## Models
All pretrained models are based on a simple U-net architecture and stored in [weights](weights) folder. The PyTorch implementation has been taken from this [repo](https://github.com/jvanvugt/pytorch-unet). Some changes are made and experimented with, see paper.
If you want to run predictions use [save_prediction.py](save_prediction.py) and make sure the designated folder for the predictions exists.

## Results
![](figures/predictions-id3-val3.png?raw=true)

Scores on the test set: [link to Cityscapes benchmark](https://www.cityscapes-dataset.com/anonymous-results/?id=9737cb5272895b3ac9e29074c4860d5746ea38b45138737fb0e27583c06f9fc5) 

categories     | IoU       
---------------|----------
construction   | 66.7421
flat           | 89.7289
human          | 5.89255
nature         | 74.5018
object         | 3.63665
sky            | 78.8369
vehicle        | 48.5196
---------------|----------
**Score Average**  | 52.5512
