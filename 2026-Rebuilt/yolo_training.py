#!/usr/bin/env python
# coding: utf-8

# In[1]:


import subprocess
import sys
import os

libraries = ["ultralytics"]

for lib in libraries:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", lib])


# In[2]:


CLUSTER = False
if not CLUSTER:
    HOME = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2026-Rebuilt"
else:
    HOME = "/home/rhayrapetyan/2025-Coral_Detection_Training"
    os.makedirs(f"{HOME}/robot_detection/runs/train", exist_ok=True)


# In[3]:


from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
if CLUSTER:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    results = model.train(
        data=f"{HOME}/FRC 2024.v4i.yolov12/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="robot-detection-model-v1",
        device=[0,1],
        workers=2,
        project=f"{HOME}/robot_detection/runs/train"
    )    
else:
    results = model.train(
        data=f"/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2026-Rebuilt/merged/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="robot-detection-model-v1",
        workers=8,
        project=f"{HOME}/robot_detection/runs/train"
    )

metrics = model.val(split="test")


# In[5]:


from IPython.display import Image, display
import os
display(Image(filename=f'/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/runs/detect/val/confusion_matrix.png', width=600))
display(Image(filename=f'/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/runs/detect/val/BoxPR_curve.png', width=600))
display(Image(filename=f'/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/runs/detect/val/val_batch0_pred.jpg', width=600))
display(Image(filename=f'/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/runs/detect/val/val_batch1_pred.jpg', width=600))
display(Image(filename=f'/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/runs/detect/val/val_batch2_pred.jpg', width=600))


# In[ ]:




