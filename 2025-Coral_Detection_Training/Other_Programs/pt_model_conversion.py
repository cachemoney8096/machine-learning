import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import sys
import os

class TransposeWrapper(nn.Module):
    """
    Wraps a PyTorch model and adds transpose to the output.
    Changes output from [1, 6, 8400] to [1, 8400, 6]
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
    
    def forward(self, x):
        # Run original model
        output = self.model(x)
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            # YOLOv8 returns a list, find the detection tensor
            for item in output:
                if isinstance(item, torch.Tensor) and len(item.shape) == 3:
                    # This should be the detection tensor [batch, channels, detections]
                    output = item
                    break
            
            # If we still have a list, take the first tensor
            if isinstance(output, (list, tuple)):
                output = output[0]
        
        # Transpose from [batch, 6, 8400] to [batch, 8400, 6]
        if len(output.shape) == 3:
            output = output.permute(0, 2, 1)
        
        return output


def load_ultralytics_model(pt_model_path):
    """
    Load a YOLOv8/Ultralytics model properly.
    """
    print("=" * 80)
    print("Loading Ultralytics/YOLOv8 Model")
    print("=" * 80)
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics library found")
        
        print(f"\n📁 Loading: {pt_model_path}")
        
        # Load using Ultralytics
        yolo_model = YOLO(pt_model_path)
        
        # Get the underlying PyTorch model
        model = yolo_model.model
        model.eval()
        
        print("✓ Model loaded successfully!")
        
        # Test the model
        print("\n🧪 Testing model...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if isinstance(output, (list, tuple)):
            print(f"✓ Model output (list/tuple with {len(output)} items)")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"   Output[{i}] shape: {o.shape}")
                else:
                    print(f"   Output[{i}] type: {type(o)}")
            # Use the first tensor output
            output = output[0] if isinstance(output[0], torch.Tensor) else output[1]
            print(f"✓ Using output shape: {output.shape}")
        else:
            print(f"✓ Model output shape: {output.shape}")
        
        return model
        
    except ImportError:
        print("❌ Ultralytics not installed")
        print("   Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_pytorch_model_safe(pt_model_path):
    """
    Load PyTorch model with weights_only=False for Ultralytics models.
    """
    print("=" * 80)
    print("Loading PyTorch Model (Safe Mode)")
    print("=" * 80)
    
    print(f"\n📁 Loading: {pt_model_path}")
    
    try:
        # For Ultralytics models, we need weights_only=False
        checkpoint = torch.load(pt_model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
                print("✓ Loaded from checkpoint['model']")
            elif 'ema' in checkpoint:
                model = checkpoint['ema'].ema
                print("✓ Loaded from checkpoint['ema']")
            else:
                print(f"⚠️  Checkpoint keys: {checkpoint.keys()}")
                return None
        else:
            model = checkpoint
            print("✓ Loaded model directly")
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
            
        # Fuse model for inference (optional but recommended)
        if hasattr(model, 'fuse'):
            print("🔧 Fusing model layers...")
            model.fuse()
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def add_transpose_to_model(model, output_pt_path=None):
    """
    Wrap model with transpose layer.
    """
    print("\n" + "=" * 80)
    print("Adding Transpose Layer")
    print("=" * 80)
    
    # Test the original model
    print("\n🧪 Testing original model...")
    dummy_input = torch.randn(1, 3, 640, 640)
    
    try:
        with torch.no_grad():
            original_output = model(dummy_input)
        
        if isinstance(original_output, (list, tuple)):
            print(f"✓ Original output (list/tuple with {len(original_output)} items)")
            for i, o in enumerate(original_output):
                if isinstance(o, torch.Tensor):
                    print(f"   Output[{i}] shape: {o.shape}")
                else:
                    print(f"   Output[{i}] type: {type(o)}")
            
            # Find the detection tensor (usually has 3 dimensions)
            for item in original_output:
                if isinstance(item, torch.Tensor) and len(item.shape) == 3:
                    original_output = item
                    break
            
            if isinstance(original_output, (list, tuple)):
                original_output = original_output[0]
            
            print(f"✓ Using output shape: {original_output.shape}")
        else:
            print(f"✓ Original output shape: {original_output.shape}")
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return None
    
    # Wrap with transpose
    print("\n🔧 Wrapping with transpose...")
    wrapped_model = TransposeWrapper(model)
    wrapped_model.eval()
    
    # Test wrapped model
    with torch.no_grad():
        new_output = wrapped_model(dummy_input)
    print(f"✓ New output shape: {new_output.shape}")
    
    # Save the wrapped model
    if output_pt_path:
        print(f"\n💾 Saving wrapped model to: {output_pt_path}")
        torch.save(wrapped_model, output_pt_path)
        print("✓ Saved!")
    
    return wrapped_model


def export_to_onnx(model, output_onnx_path, input_shape=(1, 3, 640, 640)):
    """
    Export PyTorch model to ONNX format.
    """
    print("\n" + "=" * 80)
    print("Exporting to ONNX")
    print("=" * 80)
    
    dummy_input = torch.randn(*input_shape)
    
    print(f"\n📤 Input shape: {input_shape}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
        print(f"✓ Saved ONNX model to: {output_onnx_path}")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verified!")
        
        return True
    except Exception as e:
        print(f"❌ Error exporting to ONNX: {e}")
        return False


def convert_onnx_to_tflite(onnx_path, output_tflite_path):
    """
    Convert ONNX model to TFLite using onnx2tf.
    """
    print("\n" + "=" * 80)
    print("Converting ONNX to TFLite")
    print("=" * 80)
    
    try:
        import onnx2tf
        
        print(f"\n📁 Loading ONNX: {onnx_path}")
        
        # Create output directory
        saved_model_dir = output_tflite_path.replace('.tflite', '_saved_model')
        os.makedirs(saved_model_dir, exist_ok=True)
        
        # Convert ONNX to TensorFlow SavedModel
        print("🔄 Converting to TensorFlow SavedModel...")
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            non_verbose=True
        )
        
        # Convert SavedModel to TFLite
        print("🔄 Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        # Optional optimizations (comment out if causing issues)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save
        print(f"💾 Saving TFLite model to: {output_tflite_path}")
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print("✓ Conversion complete!")
        
        # Verify
        verify_tflite_model(output_tflite_path)
        
        return True
        
    except ImportError:
        print("❌ onnx2tf not installed")
        print("   Install with: pip install onnx2tf")
        return False
    except Exception as e:
        print(f"❌ Error converting: {e}")
        return False


def verify_tflite_model(tflite_path):
    """
    Verify the TFLite model shape.
    """
    print("\n" + "=" * 80)
    print("Verifying TFLite Model")
    print("=" * 80)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n✓ Input shape:  {input_details[0]['shape'].tolist()}")
        print(f"✓ Output shape: {output_details[0]['shape'].tolist()}")
        
        # Test inference
        print("\n🧪 Testing inference...")
        dummy_input = np.random.randn(*input_details[0]['shape']).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✓ Inference successful! Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        return False


def main():
    print("\n🔧 YOLOv8/Ultralytics Model Transpose & Export Tool")
    print("=" * 80)
    
    pt_model_path = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v16/weights/best.pt'
    output_tflite_path = "Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/output_transpose_only.tflite"

    print(f"\n📁 Input:  {pt_model_path}")
    print(f"💾 Output: {output_tflite_path}\n")
    
    # Step 1: Load model
    print("Choose loading method:")
    print("1. Load using Ultralytics library (recommended for YOLOv8)")
    print("2. Load directly with torch.load")
    
    choice = input("\nEnter choice (1-2) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        model = load_ultralytics_model(pt_model_path)
    else:
        model = load_pytorch_model_safe(pt_model_path)
    
    if model is None:
        print("\n❌ Failed to load model")
        sys.exit(1)
    
    # Step 2: Add transpose
    wrapped_pt_path = pt_model_path.replace('.pt', '_transposed.pt')
    wrapped_model = add_transpose_to_model(model, wrapped_pt_path)
    
    if wrapped_model is None:
        print("\n❌ Failed to wrap model")
        sys.exit(1)
    
    # Step 3: Export
    print("\n" + "=" * 80)
    print("Export Options")
    print("=" * 80)
    print("1. Export to ONNX and TFLite (full pipeline)")
    print("2. Export to ONNX only")
    print("3. Skip export (save wrapped .pt only)")
    
    export_choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if export_choice in ["1", "2"]:
        # Get input shape
        input_shape_str = input("\nInput shape [default: 1,3,640,640]: ").strip()
        if input_shape_str:
            input_shape = tuple(map(int, input_shape_str.split(',')))
        else:
            input_shape = (1, 3, 640, 640)
        
        # Export to ONNX
        onnx_path = output_tflite_path.replace('.tflite', '.onnx')
        if export_to_onnx(wrapped_model, onnx_path, input_shape):
            if export_choice == "1":
                # Convert to TFLite
                convert_onnx_to_tflite(onnx_path, output_tflite_path)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Wrapped PyTorch model: {wrapped_pt_path}")
    if export_choice in ["1", "2"]:
        print(f"✓ ONNX model: {onnx_path}")
    if export_choice == "1":
        print(f"✓ TFLite model: {output_tflite_path}")


if __name__ == "__main__":
    main()