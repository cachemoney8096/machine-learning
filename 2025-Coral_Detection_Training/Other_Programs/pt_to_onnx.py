"""
Re-export a PyTorch model to ONNX with a compatible opset version
"""

# Method 1: Using Ultralytics YOLO (if you have YOLOv8)
def export_yolov8_to_onnx(pt_model_path, opset_version=17):
    """
    Export YOLOv8 model to ONNX with specified opset
    """
    try:
        from ultralytics import YOLO
        
        print(f"Loading model from {pt_model_path}...")
        model = YOLO(pt_model_path)
        
        print(f"Exporting to ONNX with opset {opset_version}...")
        model.export(format='onnx', opset=opset_version, simplify=True)
        
        print("Export complete!")
        print(f"ONNX model saved as: {pt_model_path.replace('.pt', '.onnx')}")
        
    except ImportError:
        print("ultralytics not installed. Install with: pip install ultralytics")
    except Exception as e:
        print(f"Error: {e}")


# Method 2: Using YOLOv5 (if you have YOLOv5)
def export_yolov5_to_onnx(pt_model_path, opset_version=17):
    """
    Export YOLOv5 model to ONNX with specified opset
    """
    try:
        import torch
        
        print(f"Loading model from {pt_model_path}...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_model_path)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        output_path = pt_model_path.replace('.pt', f'_opset{opset_version}.onnx')
        
        print(f"Exporting to ONNX with opset {opset_version}...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
        
        print("Export complete!")
        print(f"ONNX model saved as: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")


# Method 3: Using torch.load directly (most flexible)
def export_torch_model_to_onnx(pt_model_path, opset_version=17, img_size=640):
    """
    Export any PyTorch model to ONNX with specified opset
    """
    try:
        import torch
        
        print(f"Loading model from {pt_model_path}...")
        
        # Try loading the model
        checkpoint = torch.load(pt_model_path, map_location='cpu')
        
        # Extract model from checkpoint
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
            elif 'ema' in checkpoint:
                model = checkpoint['ema']
            else:
                print("Could not find model in checkpoint")
                return
        else:
            model = checkpoint
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        if hasattr(model, 'float'):
            model.float()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, img_size, img_size)
        
        output_path = pt_model_path.replace('.pt', f'_opset{opset_version}.onnx')
        
        print(f"Exporting to ONNX with opset {opset_version}...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
        
        print("Export complete!")
        print(f"ONNX model saved as: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Update this path to your .pt model
    pt_model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best.pt"
    
    # Try opset versions in order of preference: 17, 16, 14, 13, 12
    # Opset 17 is recommended for good compatibility
    opset_version = 17
    
    print("=" * 60)
    print("Attempting YOLOv8/Ultralytics export...")
    print("=" * 60)
    export_yolov8_to_onnx(pt_model_path, opset_version)
    
    # If the above doesn't work, uncomment one of these:
    
    # print("\n" + "=" * 60)
    # print("Attempting YOLOv5 export...")
    # print("=" * 60)
    # export_yolov5_to_onnx(pt_model_path, opset_version)
    
    # print("\n" + "=" * 60)
    # print("Attempting direct PyTorch export...")
    # print("=" * 60)
    # export_torch_model_to_onnx(pt_model_path, opset_version)