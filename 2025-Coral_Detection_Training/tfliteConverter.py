import os
import torch
from ultralytics import YOLO

model = YOLO("/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/robot_detection/runs/train/coral-detection-model-v15/weights/best.pt")  # Load a model
model.export(format="tflite", int8=True)
