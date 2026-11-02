#!/usr/bin/env python3
"""
Broiler Density Detection using KDTree Spatial Clustering

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
# [Colab Import]: from google.colab.patches import cv2_imshow
def cv2_imshow(img):
    """Fallback for google.colab.patches.cv2_imshow"""
    cv2.imshow("Window", img)
    cv2.waitKey(1)
from random import randint
# [Colab Import]: from google.colab import drive
# [Colab Drive Mount]: drive.mount('/content/drive')
import os
os.getcwd()
os.chdir('/content/drive/MyDrive/Colab Notebooks/ICCKE2022_Challenge/yolo-v4-tf.keras')
from models import Yolov4
from config import yolo_config
import scipy.spatial as spatial
######### Libraries and Directories

######### Configuring the Broiler Detection
#### Directory of Trainig Classes
class_name_path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/Classes/classes.txt'
#### Directory of Yolov4 Weights for Fine Tuning
weight_path = '/content/drive/My Drive/Colab Notebooks/darknet/yolov4.conv.137'
######### Configuring the Broiler Detection

###### Test Images for Dense Broilers Detection
img_path1 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00301.png'
img_path2 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00601.png'
img_path3 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00801.png'
img_path4 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame00901.png'
img_path5 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01201.png'
img_path6 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01001.png'
img_path7 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01501.png'
img_path8 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame01701.png'
img_path9 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame02901.png'
img_path10 = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/img_test/frame03901.png'
###### Test Images for Dense Broilers Detection

###### Loading the Stored Model for Broiler Detection
path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/final_training'
my_model = Yolov4(weight_path=weight_path,
               class_name_path=class_name_path)
my_model.load_model(path)
###### Loading the Stored Model for Broiler Detection

######## Broiler Density Detection Based on KDTree
def my_Density_detection(img_input):
  img=cv2.imread(img_input)
  ####### Detected Broilers
  out,my_colors= my_model.predict_img(img, plot_img=True)
  out_np = out.to_numpy()
  bb_xy = out_np[:,0:2]
  bb_wh = out_np[:,6:8]
  bb_xy = bb_xy.astype(int)
  bb_wh = bb_wh.astype(int)
  ####### Detected Broilers
  ####### Finding the nearest BB in a fixed Radius
  points=bb_xy
  tree = spatial.KDTree(np.array(points))
  radius = 100
  neighbors = tree.query_ball_tree(tree, radius)
  np_neighbors = np.array(neighbors)
  Density = np.array(list(map(len, neighbors)))
  index=np.where(Density>=6)
  ####### Finding the nearest BB in a fixed Radius
  ####### Showing the Results
  if len(index[0])>0:
    for i in range(len(index[0])):
      temp_rect=np_neighbors[index][i]
      for j in range(len(temp_rect)):
        cv2.rectangle(img, (points[temp_rect[j]][0], points[temp_rect[j]][1]), (points[temp_rect[j]][0] + bb_wh[temp_rect[j]][0], points[temp_rect[j]][1] + bb_wh[temp_rect[j]][1]), (0, 145, 255), 1)
        cv2.putText(img, 'High Dense', (points[temp_rect[j]][0], points[temp_rect[j]][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 145, 255), 2)
  cv2_imshow(img)
  ####### Showing the Results
  print('Density (number of broilers in 100 pixel):',Density)
######## Broiler Density Detection Based on KDTree

######## Results of Test Images
my_Density_detection(img_path1)
my_Density_detection(img_path2)
my_Density_detection(img_path3)
my_Density_detection(img_path4)
my_Density_detection(img_path5)
my_Density_detection(img_path6)
my_Density_detection(img_path7)
my_Density_detection(img_path8)
my_Density_detection(img_path9)
my_Density_detection(img_path10)
######## Results of Test Images


