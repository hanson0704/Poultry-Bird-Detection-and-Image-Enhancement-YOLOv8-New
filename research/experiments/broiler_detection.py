#!/usr/bin/env python3
"""
Broiler Detection using YOLOv4 and Data Generator

Historical experiment script converted from Colab Jupyter Notebook.
Original research paper: Chicken Behavior Analysis (ICCKE 2022 Challenge).
Note: Interactive Colab shell commands and mounts are converted to comments.
"""

######### Libraries and Directories
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import np_utils
import cv2 as cv
import cv2
from random import randint
# [Colab Import]: from google.colab import drive
# [Colab Drive Mount]: drive.mount('/content/drive')
import os
os.getcwd()
os.chdir('/content/drive/MyDrive/Colab Notebooks/ICCKE2022_Challenge/yolo-v4-tf.keras')
from utils import DataGenerator, read_annotation_lines
from models import Yolov4
from config import yolo_config
######### Libraries and Directories

######### Converting the xml Annotation file to txt Annotation file
#!python xml_to_txt.py
######### Converting the xml Annotation file to txt Annotation file

######### Setting Yolov4 for Transferring Learning and Fine Tuning
#### Directory of Annotations
train_lines, val_lines = read_annotation_lines('/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/Annotation/anno3.txt', test_size=0.1)
#### Directory of Trainig Images
FOLDER_PATH = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img'
#### Directory of Trainig Classes
class_name_path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/Classes/classes.txt'
#### Generating the Training Data for Yolov4
data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)
#### Directory of Yolov4 Weights for Fine Tuning
weight_path = '/content/drive/My Drive/Colab Notebooks/darknet/yolov4.conv.137'
######### Setting Yolov4 for Transferring Learning and Fine Tuning

###### Creating a new model and Start Training
model = Yolov4(weight_path=weight_path,class_name_path=class_name_path)
h = model.fit(data_gen_train,
          initial_epoch=0,
          epochs=1000,
          val_data_gen=data_gen_val,
          callbacks=[])
#model.save_model('/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/final_training')
###### Creating a new model and Start Training
'''
###### Loading the Stored Model
path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/final_training'
my_model = Yolov4(weight_path=weight_path,
               class_name_path=class_name_path)
my_model.load_model(path)
###### Loading the Stored Model
'''
###### Testing the Trained Model For Broiler Detection
img_path1 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00097.png'
img_path2 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00201.png'
img_path3 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00289.png'
img_path4 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00301.png'
img_path5 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00601.png'
img_path6 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00801.png'
img_path7 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00901.png'
img_path8 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01201.png'
img_path9 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01001.png'
img_path10 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01301.png'
img_path11 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01501.png'
img_path12 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01701.png'
img_path13 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame02601.png'
img_path14 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame02901.png'
img_path15 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame03901.png'
img_path16 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame04101.png'
my_model.predict(img_path1)
my_model.predict(img_path2)
my_model.predict(img_path3)
my_model.predict(img_path4)
my_model.predict(img_path5)
my_model.predict(img_path6)
my_model.predict(img_path7)
my_model.predict(img_path8)
my_model.predict(img_path9)
my_model.predict(img_path10)
my_model.predict(img_path11)
my_model.predict(img_path12)
my_model.predict(img_path13)
my_model.predict(img_path14)
my_model.predict(img_path15)
my_model.predict(img_path16)
###### Testing the Trained Model For Broiler Detection


