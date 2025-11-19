from ultralytics import YOLO

# Hardcoded paths
pt_path = r"C:\\Users\\Student Robotic\\Desktop\\ruben\\machine-learning\\2025-Coral_Detection_Training\\coral-detection-model-v1\\best.pt"
onnx_path = pt_path.replace(".pt", ".onnx")

# Load YOLO model safely
model = YOLO(pt_path)

# Export to ONNX
model.export(format="onnx", opset=12, dynamic=True, simplify=True, imgsz=640)
print(f"ONNX exported to {onnx_path}")
