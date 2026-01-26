import torch
import tensorflow as tf
import numpy as np
import onnx
import tf2onnx
import os
import tempfile

class PyTorchToTFLiteConverter:
    def __init__(self, pt_model_path, input_shape, model_type='yolo'):
        """
        Initialize converter
        
        Args:
            pt_model_path: Path to .pt PyTorch model file
            input_shape: Input shape as tuple (batch, channels, height, width) for PyTorch
                        Example: (1, 3, 640, 640)
            model_type: 'yolo' or 'custom' - helps with specific model handling
        """
        self.pt_model_path = pt_model_path
        self.input_shape = input_shape
        self.model_type = model_type
        
    def convert_to_quantized_tflite(self, output_path, representative_data_dir=None, 
                                     num_calibration_samples=100):
        """
        Complete pipeline: PyTorch -> ONNX -> TensorFlow -> Quantized TFLite
        
        Args:
            output_path: Path for output .tflite file
            representative_data_dir: Directory with calibration images (optional but recommended)
            num_calibration_samples: Number of samples for quantization calibration
        """
        print("="*70)
        print("PyTorch to Quantized TFLite Conversion Pipeline")
        print("="*70)
        
        # Step 1: Load PyTorch model
        print("\n[1/5] Loading PyTorch model...")
        model = self._load_pytorch_model()
        
        # Step 2: Export to ONNX
        print("\n[2/5] Converting PyTorch -> ONNX...")
        onnx_path = self.pt_model_path.replace('.pt', '.onnx')
        self._export_to_onnx(model, onnx_path)
        
        # Step 3: Convert ONNX to TensorFlow SavedModel using tf2onnx
        print("\n[3/5] Converting ONNX -> TensorFlow SavedModel...")
        saved_model_dir = self.pt_model_path.replace('.pt', '_saved_model')
        self._onnx_to_saved_model_tf2onnx(onnx_path, saved_model_dir)
        
        # Step 4: Convert to TFLite with quantization
        print("\n[4/5] Converting TensorFlow -> Quantized TFLite (UINT8)...")
        self._saved_model_to_quantized_tflite(
            saved_model_dir, 
            output_path,
            representative_data_dir,
            num_calibration_samples
        )
        
        # Step 5: Verify
        print("\n[5/5] Verifying quantization...")
        self._verify_tflite(output_path)
        
        print("\n" + "="*70)
        print(f"✅ SUCCESS! Quantized TFLite model saved to: {output_path}")
        print("="*70)
        
        return output_path
    
    def _load_pytorch_model(self):
        """Load PyTorch model from .pt file"""
        # Load the model - need weights_only=False for YOLOv5/Ultralytics models
        try:
            checkpoint = torch.load(self.pt_model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"   Error loading with torch.load: {e}")
            print("   Trying alternative loading method with Ultralytics...")
            
            # Try loading with ultralytics if available
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(self.pt_model_path)
                checkpoint = yolo_model.model
                print("   ✓ Loaded using Ultralytics YOLO")
            except ImportError:
                print("   ERROR: ultralytics not installed. Install with: pip install ultralytics")
                raise
            except Exception as e2:
                print(f"   ERROR: Could not load model: {e2}")
                raise
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model = checkpoint['model']
                # Handle YOLOv5/v8 format
                if hasattr(model, 'float'):
                    model = model.float()
                if hasattr(model, 'fuse'):
                    try:
                        model.fuse()
                        print("   ✓ Model fused for inference")
                    except:
                        print("   ⚠ Could not fuse model, continuing anyway...")
            elif 'ema' in checkpoint:
                model = checkpoint['ema']
                if hasattr(model, 'float'):
                    model = model.float()
            elif 'state_dict' in checkpoint:
                print("ERROR: Checkpoint contains only state_dict.")
                print("Please export your model with the full model, not just state_dict.")
                raise ValueError("Cannot load model from state_dict alone. Need full model.")
            else:
                # Try to use the dict itself
                model = checkpoint
        else:
            model = checkpoint
        
        # Set to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        if hasattr(model, 'float'):
            model.float()
        
        # Disable gradients
        try:
            for param in model.parameters():
                param.requires_grad = False
        except:
            pass
        
        print(f"   ✓ Model loaded: {type(model)}")
        return model
    
    def _export_to_onnx(self, model, onnx_path):
        """Export PyTorch model to ONNX format"""
        # Create dummy input
        dummy_input = torch.randn(self.input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,  # Use opset 12 for better compatibility
            do_constant_folding=True,
            input_names=['images'],  # Common for YOLO models
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
        
        print(f"   ✓ ONNX model saved: {onnx_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model verified")
    
    def _onnx_to_saved_model_tf2onnx(self, onnx_path, saved_model_dir):
        """Convert ONNX to TensorFlow SavedModel using tf2onnx"""
        import subprocess
        
        # Remove existing saved model if present
        if os.path.exists(saved_model_dir):
            import shutil
            shutil.rmtree(saved_model_dir)
        
        # Use tf2onnx command line tool (more reliable than Python API)
        cmd = [
            'python', '-m', 'tf2onnx.convert',
            '--onnx', onnx_path,
            '--output', saved_model_dir,
            '--saved-model'
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("   ERROR: tf2onnx conversion failed")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            
            # Try alternative method using Python API
            print("   Trying alternative conversion method...")
            self._onnx_to_saved_model_python_api(onnx_path, saved_model_dir)
        else:
            print(f"   ✓ SavedModel saved: {saved_model_dir}")
    
    def _onnx_to_saved_model_python_api(self, onnx_path, saved_model_dir):
        """Alternative: Use tf2onnx Python API"""
        import tf2onnx
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert using tf2onnx
        tf_rep = tf2onnx.convert.from_onnx(
            onnx_model,
            output_path=saved_model_dir
        )
        
        print(f"   ✓ SavedModel saved: {saved_model_dir}")
    
    def _saved_model_to_quantized_tflite(self, saved_model_dir, output_path,
                                          representative_data_dir, num_samples):
        """Convert SavedModel to fully quantized TFLite"""
        # Load SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Create representative dataset for quantization calibration
        if representative_data_dir:
            print(f"   Using calibration images from: {representative_data_dir}")
            representative_dataset = self._create_representative_dataset(
                representative_data_dir, num_samples
            )
        else:
            print("   WARNING: No calibration data provided. Using random data.")
            print("   For best results, provide real images with representative_data_dir")
            representative_dataset = self._create_random_representative_dataset(num_samples)
        
        converter.representative_dataset = representative_dataset
        
        # Force full integer quantization (UINT8 inputs and outputs)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Allow custom ops if needed (for some YOLO models)
        # converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
        
        # Convert
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"   ERROR during conversion: {e}")
            print("   Trying with relaxed quantization settings...")
            
            # Try with fallback to float for unsupported ops
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"   ✓ Quantized TFLite saved: {output_path}")
        print(f"   ✓ Model size: {len(tflite_model) / (1024*1024):.2f} MB")
    
    def _create_representative_dataset(self, data_dir, num_samples):
        """Create representative dataset from real images"""
        import glob
        from PIL import Image
        
        # Get image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(data_dir, ext)))
            image_files.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        if not image_files:
            print(f"   WARNING: No images found in {data_dir}")
            return self._create_random_representative_dataset(num_samples)
        
        # Limit to num_samples
        image_files = image_files[:num_samples]
        print(f"   Found {len(image_files)} calibration images")
        
        # PyTorch input shape is (B, C, H, W), need (H, W) for resize
        _, _, height, width = self.input_shape
        
        def representative_dataset_gen():
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((width, height))
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalize to [0, 1]
                    img_array = img_array / 255.0
                    
                    # Add batch dimension: (H, W, C) -> (1, H, W, C) for TensorFlow
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    yield [img_array]
                except Exception as e:
                    print(f"   Warning: Could not load {img_path}: {e}")
                    continue
        
        return representative_dataset_gen
    
    def _create_random_representative_dataset(self, num_samples):
        """Create representative dataset from random data"""
        # Convert PyTorch shape (B, C, H, W) to TensorFlow shape (B, H, W, C)
        batch, channels, height, width = self.input_shape
        tf_shape = (batch, height, width, channels)
        
        def representative_dataset_gen():
            for _ in range(num_samples):
                data = np.random.uniform(0.0, 1.0, tf_shape).astype(np.float32)
                yield [data]
        
        return representative_dataset_gen
    
    def _verify_tflite(self, tflite_path):
        """Verify the TFLite model quantization"""
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n   INPUT DETAILS:")
        for inp in input_details:
            print(f"      Shape: {inp['shape']}, Dtype: {inp['dtype']}")
            if inp['dtype'] in [np.uint8, np.int8]:
                print(f"      ✓ Input is quantized to {inp['dtype']}")
        
        print("\n   OUTPUT DETAILS:")
        for out in output_details:
            print(f"      Shape: {out['shape']}, Dtype: {out['dtype']}")
            if out['dtype'] in [np.uint8, np.int8]:
                print(f"      ✓ Output is quantized to {out['dtype']}")
        
        # Check if fully quantized
        input_quantized = all(inp['dtype'] in [np.uint8, np.int8] for inp in input_details)
        output_quantized = all(out['dtype'] in [np.uint8, np.int8] for out in output_details)
        
        if input_quantized and output_quantized:
            print("\n   ✅ Model is FULLY QUANTIZED (INT8/UINT8)")
        else:
            print("\n   ⚠️ Model is NOT fully quantized")
            if not input_quantized:
                print("      - Some inputs are not quantized")
            if not output_quantized:
                print("      - Some outputs are not quantized")


# Simple CLI interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch .pt to Quantized TFLite')
    parser.add_argument('--input', required=True, help='Input .pt model path')
    parser.add_argument('--output', required=True, help='Output .tflite path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--calibration-data', help='Directory with calibration images')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of calibration samples')
    parser.add_argument('--model-type', default='yolo', choices=['yolo', 'custom'], help='Model type')
    
    args = parser.parse_args()
    
    # Define input shape (batch, channels, height, width) for PyTorch
    input_shape = (1, 3, args.input_size, args.input_size)
    
    # Create converter
    converter = PyTorchToTFLiteConverter(
        pt_model_path=args.input,
        input_shape=input_shape,
        model_type=args.model_type
    )
    
    # Convert
    converter.convert_to_quantized_tflite(
        output_path=args.output,
        representative_data_dir=args.calibration_data,
        num_calibration_samples=args.num_samples
    )


if __name__ == "__main__":
    import sys
    
    # Check if arguments provided
    if len(sys.argv) > 1:
        # Run CLI
        main()
    else:
        # Show help
        print("="*70)
        print("PyTorch to Quantized TFLite Converter")
        print("="*70)
        print("\nCommand line usage:")
        print("python script.py --input model.pt --output model_uint8.tflite --input-size 640 --calibration-data ./images")
        print("\nPython code usage:")
        print("""
converter = PyTorchToTFLiteConverter(
    pt_model_path='model.pt',
    input_shape=(1, 3, 640, 640)
)
converter.convert_to_quantized_tflite(
    output_path='model_uint8.tflite',
    representative_data_dir='./calibration_images',
    num_calibration_samples=100
)
    """)
        print("\nRequired packages:")
        print("pip install torch tensorflow onnx tf2onnx pillow numpy")
        print("="*70)