import torch
import torch.nn as nn

from ultralytics import YOLO

model = YOLO('/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v16/weights/best.pt')

# Export with different quantization options:

# 1. INT8 TFLite (smallest, needs calibration data)
model.export(format='tflite', int8=True, data='/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/2025 REEFSCAPE.v2i.yolov12/data.yaml')

# 2. FP16 TFLite (balanced)
# model.export(format='tflite', half=True)

# 3. Dynamic range quantization
# model.export(format='tflite')