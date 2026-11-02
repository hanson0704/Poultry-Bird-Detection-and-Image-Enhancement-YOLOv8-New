#!/usr/bin/env python3
"""
Broiler Multi-Object Tracking using CentroidTracker

Historical experiment script converted from Colab Jupyter Notebook.
Original research paper: Chicken Behavior Analysis (ICCKE 2022 Challenge).
Note: Interactive Colab shell commands and mounts are converted to comments.
"""

######### System Configuration Output
# [Notebook command]: !cat /proc/cpuinfo
# [Notebook command]: !nvidia-smi
######### System Configuration Output

######### Libraries and Directories
import os
# [Colab Import]: from google.colab import drive
# [Colab Drive Mount]: drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/Colab Notebooks/ICCKE2022_Challenge')
# [Notebook command]: %cd multi-object-tracker
# [Notebook command]: !pip install -r requirements.txt
# [Notebook command]: !pip install -e
from motrackers import CentroidTracker
from motrackers.utils import draw_tracks
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from random import randint
# [Colab Import]: from google.colab.patches import cv2_imshow
def cv2_imshow(img):
    """Fallback for google.colab.patches.cv2_imshow"""
    cv2.imshow("Window", img)
    cv2.waitKey(1)
os.chdir('/content/drive/MyDrive/Colab Notebooks/ICCKE2022_Challenge/yolo-v4-tf.keras')
######## Libraries and Directories

######## Loading The Trained Yolov4 Model for Detection
class_name_path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/Classes/classes.txt'
weight_path = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/darknet/yolov4.conv.13'
path = '/content/drive/MyDrive/Colab Notebooks/ICCKE2022_Challenge/final_training'
from models import Yolov4
model = Yolov4(weight_path=weight_path,class_name_path=class_name_path)
model.load_model(path)
######## Loading The Trained Yolov4 Model for Detection

######## Directory for Video Test
videoPath = '/content/drive/My Drive/Colab Notebooks/ICCKE2022_Challenge/video_test/2.mp4'
######## Directory for Video Test

######## Detection Function, Input:Frame , Output: Bounding Boxes with Colors, Score of Detections, Classes of Detections
def my_detection(my_frame):
  t1 = time.perf_counter()
  out,my_colors= model.predict_img(my_frame,plot_img=False)
  out_np = out.to_numpy()
  bb_xywh = np.append(out_np[:,0:2],  out_np[:,6:8], axis=1)
  bb_score = out_np[:,5]
  bb_class = out_np[:,0]
  bb_xywh = bb_xywh.astype(int)
  bb_score = bb_score.astype(int)
  bb_class = bb_class.astype(int)
  t2 = time.perf_counter()
  print('Detection Time Overhead:',t2-t1)
  return my_colors, bb_xywh, bb_score, bb_class
######## Detection Function, Input:Frame , Output: Bounding Boxes with Colors, Score of Detections, Classes of Detections

######## Setting the Tracker
tracker = CentroidTracker(max_lost=5, tracker_output_format='mot_challenge')
######## Setting the Tracker

######## Capturing the Video Input
cap = cv2.VideoCapture(videoPath)
length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
success, frame = cap.read()
frame=cv2.resize(frame, (480, 480))
######## Capturing the Video Input

######## Frames list for making output video
frames=[]
######## Frames list for making output video

######## Initializing the Detection and Tracking
my_colors,my_bbxywh, my_score, my_class = my_detection(frame)
tracks = tracker.update(my_bbxywh, my_score, my_class)
######## Initializing the Detection and Tracking

######## Video Detection and Tracking Start
for i in range(1000):
  t1 = time.perf_counter()
  print('Frame Number:',i)
  ret, frame = cap.read()
  frame=cv2.resize(frame, (480, 480))

  ######## Loading New Tracks Data
  my_colors,my_bbxywh, my_score, my_class = my_detection(frame)
  tracks = tracker.update(my_bbxywh, my_score, my_class)
  ######## Loading New Tracks Data

  ######## Data of each Tracking ID
  for trk in tracks:
    trk_id = trk[1]
    xmin = int(trk[2])
    ymin = int(trk[3])
    width = int(trk[4])
    height = int(trk[5])
    xcentroid_show, ycentroid_show = int(xmin + 0.5*width), int(ymin + 0.5*height)
    ######## Adding the Results to the Frames
    text = "ID {}".format(trk_id)
    cv2.putText(frame, text, (xcentroid_show - 10, ycentroid_show - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (50, 255, 50), 1)
    cv2.circle(frame, (xcentroid_show, ycentroid_show), 2, (0, 255, 0), -1)
    cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), (50, 255, 50), 1)
    ######## Adding the Results to the Frames

  ######## Appending the Processed Frames
  t2 = time.perf_counter()
  print('Total Detection and Tracking Time:',t2-t1)
  if (i % 7==0):
    cv2_imshow(frame)
  frames.append(frame)
  ######## Appending the Processed Frames


######## Making the Output Video
video = cv2.VideoWriter('Tracking_Result.avi', 0, 25, (480, 480))
for img in frames:
  video.write(img)
video.release()
######## Making the Output Video

######## Video Detection and Tracking Stop




