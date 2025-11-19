from ultralytics import YOLO

# Hardcoded paths - use the .pt file
pt_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v1/best.pt"

# Load the PyTorch model
model = YOLO(pt_path)

# Export directly to TFLite
model.export(format='tflite', imgsz=640)

print(f"Converted {pt_path} to TFLite")