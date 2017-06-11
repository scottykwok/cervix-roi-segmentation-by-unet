#Cervix ROI Segmentation Using U-NET
 
## Overview
This code illustrate how to segment the ROI in cervical images using U-NET. The ROI here meant to include the: Os + transformation zone + nearby tissue.

The localized ROI could be used to improve the classification of cervical types, which is the challenge in the Kaggle competition:[Intel and MobileODT Cervical Cancer Screening](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening) 



**Dependencies:**
- Keras 2
- Tensorflow or Theano
- cv2

**Reference:**
- Thanks to [Paul](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31565), who annotated the bounding box here: [Bounding boxes for Type_1](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31565). I have made some minor adjustments and added the missing images.   
- The amazing Keras UNET implementation by jocicmarko in: [ultrasound-nerve-segmentation](https://github.com/jocicmarko/ultrasound-nerve-segmentation)
- The original U-Net design:  [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


## Model
Note in this Keras2 U-NET implementations, we have:   
 - the input images in RGB 
 - the input images and masks are augmented in pairs using izip ImageDataGenerators
 - support both Tensorflow and Theano backend, and is using Keras 2


## Usage

**Data preparation:**
- Download the [data from Kaggle](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data). Unzip `trian.7z` and `test.7z` into input folder
- Run `prepare_data.py`
- Run `split_data.py`
- Note:
    - The bbox annotations were converted to Sloth json format and is included under `input/*.json`.
    - The additional data is NOT used in this training.

**Training:**
- Run `train.py`
- Todo: upload a pretrained tensorflow weight file.

**Segmentation:**
- Run `predict.py`
- The output segmentations are under:
    - input/test_roi/
    - input/train_roi/
    
   
## Results
On a GTX 1070, the training of 400 epochs took ~2 hours to complete. The best DICE coefficient is ~0.79. 

Apply this model to the 512 unseen test images, the result looks satisfactory in 96% of images.

Sample outputs:
![img/preview.jpg](img/preview.jpg)


