# Roadment

## Introduction
Semantic segmentation is the process of classifying each pixel of an image into distinct classes using deep learning. This aids in identifying regions in an image where certain objects reside.

The aim of this project is to build a web application which will identify and segment roads in aerial images.Road segmentation in aerial images is a crucial process in the workflow of target based aerial image analysis. This task focusses to mapping every pixel to a semantic label. It has various applications such as road navigation, urban infrastructure development and geographic information collection.In the last decade, a lot of research has happened related to road information extraction from aerial imagery. Still it remains a challenging task due to noise, complex Image background and occlusions.

This project will be the first attempt to build an end-to-end, user friendly platform for segmentation of roads. Moreover existing technologies such as Google Maps rely on human effort for annotation of their maps. Our project aims at automating the task of pointing out the roads in such maps, which require frequent updates and checking. 

## Contents:
1. Dataset.
2. Manipulating the data.
3. Model Architecture

## Dataset
For this project, we will use the [Massachusetts Roads Dataset](https://www.kaggle.com/balraj98/massachusetts-roads-dataset). This dataset contains aerial images, along with the target masks.The dataset contains 1171 images and respective masks. Both the masks and the images are 1500x1500 in the resolution are present in the .tiff format. 

## Manipulating the data


## Model Architecture
We used a standard fully convolutional UNet structure that receives an RGB colour image as input and generates a same-size semantic segmentation map as output. The structure of the network is an encoder-decoder network with skip connections between various feature levels of the encoder to the decoder. This enables the network to combine information from both deep abstract features and local, high-resolution information to generate the final output.

The chosen UNet structure is a proven architecture for extracting both deep abstract features and local features. It consists of three sections- the contraction, bottleneck, and expansion. The downsampling contraction section, known as the encoder, extracts feature information from the input by decreasing in spatial dimensions but increasing in feature dimensions. The bottleneck is where we have a compact representation of the input. The upsampling expansion section, known as the decoder, reduces the image in feature dimension while increasing the spatial dimensions. It uses skip connections that allow it to tap into the same-sized output of the contraction section, which allows it to use higher-level locality features in its upsampling. This is a strong architecture for image segmentation, as we must assign each pixel a class. Thus, our output must use the high-resolution information from the input image to obtain a good prediction.
