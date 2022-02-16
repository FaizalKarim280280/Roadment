import segmentation_models as sm
import numpy as np
import os
import cv2 as op
import matplotlib.pyplot as plt
import tensorflow as tf
from os import listdir
from tensorflow import keras
from keras import backend as K

class ModelPrediction:
    def __init__(self,Image_Size):
        self.IMAGE_SIZE = Image_Size

    @staticmethod
    def dice_coef(self,y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def load_model(self):
        objects = {
            'dice_coef': self.dice_coef,
            'dice_coef_loss': self.dice_coef_loss,
            'iou_score': sm.metrics.iou_score
        }

        model = keras.models.load_model(
            r'C:\Users\Asus\Desktop\RoadMent\Web\Model\model_main_loss=0.36_iou=0.47.h5',
            custom_objects=objects)
        return model
    
    def placeMaskOnImg(self,img, mask):
        color = [66, 255, 73]
        color = [i / 255.0 for i in color]
        np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
        return img
    
    def make_pred_good(self,pred):
        pred = pred[0][:, :, :]
        pred = np.repeat(pred, 3, 2)
        return pred
    
    def compute(self,model, PATH):
        img = op.imread(PATH)
        img = op.cvtColor(img, op.COLOR_BGR2RGB)

        img = img / 255.0
        img = op.resize(img, (512, 512))
        img = np.expand_dims(img, axis=0)
        img = img[:, :, :, :3]

        pred = self.make_pred_good(model(img))
        pred = self.placeMaskOnImg(img[0], pred)

        plt.axis('off')
        plt.grid(False)

        plt.imsave(r'C:/Users/Asus/Desktop/RoadMent/Web/static/Satellite Images/out_plot.png', pred)