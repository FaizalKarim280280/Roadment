# Project Title: RoadMent
Roadment is an end-to-end platform for segmentation of roads from satellite images. We have built a website, integrated with a deep learning model using Flask API. A demo is provided below:

![Alt text](https://github.com/FaizalKarim280280/Roadment/blob/fk/plots/demo.gif)
<br><br>
![Alt text](https://github.com/FaizalKarim280280/Roadment/blob/fk/plots/pred.jpg)

# Description
Semantic segmentation is the process of classifying each pixel of an image into distinct classes using deep learning. This aids in identifying regions in an image where certain objects reside.

The aim of this project is to build a web application which will identify and segment roads in aerial images.Road segmentation in aerial images is a crucial process in the workflow of target based aerial image analysis. In the last decade, a lot of research has happened related to road information extraction from aerial imagery. Still it remains a challenging task due to noise, complex Image background and occlusions.

Existing technologies such as Google Maps rely on human effort for annotation of their maps. Our project aims at automating the task of pointing out the roads in such maps, which require frequent updates and checking. 

# Contents:
1. Dataset
2. Tools and libraries    
3. Data Preprcessing
4. Model Architecture
5. Training and evaluation
6. References

## 1. Dataset
For this project, we have used the [Massachusetts Roads Dataset](https://www.kaggle.com/balraj98/massachusetts-roads-dataset). This dataset contains aerial images, along with the target masks. The dataset contains 1171 aerial images of the state of Massachusetts and their respective masks. Both the masks and the images are 1500x1500 in resolution and are present in .tiff format. 

10% of the dataset was used for validation.

## 2. Tools and libraries
* TensorFlow framework for model building.
* Albumentations for data augmentation.
* Segmentation Models for defining custom loss function.
* Other secondary libraries mentioned in requirements.txt

## 3. Data Preprocessing
1. Images were reshaped to 512x512 dimension and normalized [0,1].
2. The training images were of very high quality (1500x1500) but for testing such high quality images are not easy to gather. So, we decided to augment the images by randomly blurring them and applying random horizontal flip. On applying more augmentation methods, we didn't observe much increase in accuracy.
3. _tf.data.Dataset_ was used for building an efficient data pipeline.


## 4. Model Architecture
1. We have used a standard fully convolutional **UNet** architecture that receives an RGB colour image as input and generates a same-size semantic segmentation map as output. The structure of the network is an encoder-decoder network with skip connections between various feature levels of the encoder to the decoder.
   
2. The model has 4 downsampling blocks and 3 upsampling blocks. 
    1. The downsampling section, known as the encoder, extracts feature information from the input by decreasing in spatial dimensions but increasing in feature dimensions.
    2. The upsampling section, known as the decoder, reduces the image in feature dimension while increasing the spatial dimensions. It uses skip connections that allow it to tap into the same-sized output of the contraction section, which allows it to use higher-level locality features in its upsampling.
    
3. This is a strong architecture for image segmentation, as we must assign each pixel a class. Thus, our output must use the high-resolution information from the input image to obtain a good prediction.

## 5. Training and evaluation
1. The model was trained for 15 epochs using a batch size of 4. We have used a smaller batch size because of memory issues.
2. Loss function used during training was [dice-coeffiencet loss](https://medium.com/ai-salon/understanding-dice-loss-for-crisp-boundary-detection-bb30c2e5f62b) and metrics used were dice coefficient and [IoU score](https://towardsdatascience.com/iou-a-better-detection-evaluation-metric-45a511185be1).
3. Adam was used as optimizer and learning rate was initially set to 5e-4 and after every 2 epochs the learning rate was reduced 1.5 times.
4. After training for 15 epochs, we obtained a training iou score of 0.497 and validation iou score of 0.415.

![Alt text](https://github.com/FaizalKarim280280/Roadment/blob/fk/plots/evaluation.png)

## 6. References
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
* [Road Segmentation in Aerial Images by Exploiting Road Vector Data](https://ieeexplore.ieee.org/document/6602035) - Jiangye Yuan, Anil M. Cheriyadat (2013)
* [Road Segmentation in SAR Satellite Images with Deep Fully-Convolutional Neural Networks](https://arxiv.org/abs/1802.01445) - Corentin Henry, Seyed Majid Azimi, Nina Merkle (2018)
* [Tensorflow](https://www.tensorflow.org/)
* [Albumentations](https://albumentations.ai/docs/)
* [Segmentation Models](https://github.com/qubvel/segmentation_models)


