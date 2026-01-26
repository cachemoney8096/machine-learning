import tensorflow as tf
import numpy as np
import json
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def analyze_tflite_model(model_path):
    """Extract detailed information from a TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    info = {
        'model_path': str(model_path),
        'input_details': [],
        'output_details': [],
        'tensor_count': len(tensor_details),
        'all_tensors': []
    }
    
    # Process input details
    for inp in input_details:
        info['input_details'].append({
            'name': inp['name'],
            'shape': inp['shape'].tolist(),
            'dtype': str(inp['dtype']),
            'index': inp['index'],
            'quantization': (float(inp['quantization'][0]), int(inp['quantization'][1])),
        })
    
    # Process output details
    for out in output_details:
        info['output_details'].append({
            'name': out['name'],
            'shape': out['shape'].tolist(),
            'dtype': str(out['dtype']),
            'index': out['index'],
            'quantization': (float(out['quantization'][0]), int(out['quantization'][1])),
        })
    
    # Process all tensors
    for tensor in tensor_details:
        info['all_tensors'].append({
            'name': tensor['name'],
            'shape': tensor['shape'].tolist(),
            'dtype': str(tensor['dtype']),
            'index': tensor['index'],
            'quantization': (float(tensor['quantization'][0]), int(tensor['quantization'][1]))
        })
    
    return info, interpreter

def compare_tensor_lists(tensors1, tensors2, label="Tensors"):
    """Compare two lists of tensors."""
    print(f"\n{'='*80}")
    print(f"{label} Comparison")
    print(f"{'='*80}")
    
    print(f"\nModel 1: {len(tensors1)} {label.lower()}")
    print(f"Model 2: {len(tensors2)} {label.lower()}")
    
    if len(tensors1) != len(tensors2):
        print(f"⚠️  DIFFERENCE: Different number of {label.lower()}!")
    
    # Compare each tensor
    for i, (t1, t2) in enumerate(zip(tensors1, tensors2)):
        print(f"\n{label[:-1]} {i}:")
        print(f"  Model 1:")
        print(f"    Name: {t1['name']}")
        print(f"    Shape: {t1['shape']}")
        print(f"    DType: {t1['dtype']}")
        print(f"    Quantization: {t1['quantization']}")
        
        print(f"  Model 2:")
        print(f"    Name: {t2['name']}")
        print(f"    Shape: {t2['shape']}")
        print(f"    DType: {t2['dtype']}")
        print(f"    Quantization: {t2['quantization']}")
        
        # Highlight differences
        differences = []
        if t1['name'] != t2['name']:
            differences.append("NAME")
        if t1['shape'] != t2['shape']:
            differences.append("SHAPE")
        if t1['dtype'] != t2['dtype']:
            differences.append("DTYPE")
        if t1['quantization'] != t2['quantization']:
            differences.append("QUANTIZATION")
        
        if differences:
            print(f"  ⚠️  DIFFERENCES: {', '.join(differences)}")
        else:
            print(f"  ✓ Identical")

def load_and_preprocess_image(image_path, target_shape, dtype):
    """Load and preprocess an image for the model."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: PIL not found. Install with: pip install Pillow")
        return None
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    print(f"  Original image size: {img.size}")
    
    # Resize to target shape (height, width)
    target_size = (target_shape[2], target_shape[1])  # (width, height)
    img_resized = img.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Convert to target dtype
    if dtype == np.float32:
        # Normalize to 0-1 range
        img_array = img_array.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        img_array = img_array.astype(np.uint8)
    
    return img_array

def test_inference(interpreter, model_name, image_path=None):
    """Test inference with real image or dummy data."""
    print(f"\n{'='*80}")
    print(f"Testing Inference: {model_name}")
    print(f"{'='*80}")
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create input
    for inp in input_details:
        if image_path:
            print(f"\nLoading image: {image_path}")
            input_data = load_and_preprocess_image(image_path, inp['shape'], inp['dtype'])
            if input_data is None:
                print("Failed to load image, using random data instead")
                if inp['dtype'] == np.uint8:
                    input_data = np.random.randint(0, 255, size=inp['shape'], dtype=np.uint8)
                else:
                    input_data = np.random.randn(*inp['shape']).astype(inp['dtype'])
        else:
            print(f"\nUsing random data")
            if inp['dtype'] == np.uint8:
                input_data = np.random.randint(0, 255, size=inp['shape'], dtype=np.uint8)
            else:
                input_data = np.random.randn(*inp['shape']).astype(inp['dtype'])
        
        interpreter.set_tensor(inp['index'], input_data)
        print(f"  Input shape: {inp['shape']}, dtype: {inp['dtype']}")
    
    # Run inference
    print("  Running inference...")
    interpreter.invoke()
    print("  ✓ Inference complete")
    
    # Get outputs
    outputs = []
    for i, out in enumerate(output_details):
        output_data = interpreter.get_tensor(out['index'])
        outputs.append(output_data)
        print(f"\nOutput {i}:")
        print(f"  Name: {out['name']}")
        print(f"  Shape: {output_data.shape}")
        print(f"  DType: {output_data.dtype}")
        print(f"  Min: {output_data.min():.6f}")
        print(f"  Max: {output_data.max():.6f}")
        print(f"  Mean: {output_data.mean():.6f}")
        print(f"  Std: {output_data.std():.6f}")
    
    return outputs

def compare_output_format(outputs1, outputs2):
    """Compare output formats between two models."""
    print(f"\n{'='*80}")
    print(f"Output Format Comparison")
    print(f"{'='*80}")
    
    print(f"\nNumber of outputs:")
    print(f"  Model 1: {len(outputs1)}")
    print(f"  Model 2: {len(outputs2)}")
    
    if len(outputs1) != len(outputs2):
        print("  ⚠️  DIFFERENT number of outputs!")
    
    for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
        print(f"\nOutput {i}:")
        print(f"  Model 1 shape: {o1.shape}, dtype: {o1.dtype}")
        print(f"  Model 2 shape: {o2.shape}, dtype: {o2.dtype}")
        
        if o1.shape != o2.shape:
            print(f"  ⚠️  SHAPE DIFFERENCE!")
        if o1.dtype != o2.dtype:
            print(f"  ⚠️  DTYPE DIFFERENCE!")
        
        # Check if values are in similar ranges
        print(f"\n  Value range comparison:")
        print(f"    Model 1 - Min: {o1.min():.6f}, Max: {o1.max():.6f}, Mean: {o1.mean():.6f}")
        print(f"    Model 2 - Min: {o2.min():.6f}, Max: {o2.max():.6f}, Mean: {o2.mean():.6f}")

def print_summary(info1, info2):
    """Print a clear summary of key differences."""
    print(f"\n{'#'*80}")
    print("SUMMARY OF KEY DIFFERENCES")
    print(f"{'#'*80}\n")
    
    print("📊 INPUT DIFFERENCES:")
    print(f"  • Model 1: {info1['input_details'][0]['shape']} {info1['input_details'][0]['dtype']}")
    print(f"  • Model 2: {info2['input_details'][0]['shape']} {info2['input_details'][0]['dtype']}")
    if info1['input_details'][0]['quantization'][0] != 0:
        print(f"  • Model 1 quantization: {info1['input_details'][0]['quantization']}")
    if info2['input_details'][0]['quantization'][0] != 0:
        print(f"  • Model 2 quantization: {info2['input_details'][0]['quantization']}")
    
    print("\n📤 OUTPUT DIFFERENCES:")
    print(f"  • Model 1 has {len(info1['output_details'])} output(s)")
    print(f"  • Model 2 has {len(info2['output_details'])} output(s)")
    
    print("\n  Model 1 outputs:")
    for i, out in enumerate(info1['output_details']):
        print(f"    [{i}] {out['name']}: shape {out['shape']}")
    
    print("\n  Model 2 outputs:")
    for i, out in enumerate(info2['output_details']):
        print(f"    [{i}] {out['name']}: shape {out['shape']}")
    
    print("\n🔍 INTERPRETATION:")
    if len(info2['output_details']) == 4 and 'PostProcess' in info2['output_details'][0]['name']:
        print("  • Model 2 appears to have post-processing built-in")
        print("  • Output 0: Detection boxes [batch, num_detections, 4]")
        print("  • Output 1: Detection classes [batch, num_detections]")
        print("  • Output 2: Detection scores [batch, num_detections]")
        print("  • Output 3: Number of detections [batch]")
    
    if len(info1['output_details']) == 1 and info1['output_details'][0]['shape'][2] > 1000:
        print("  • Model 1 appears to output raw predictions")
        print("  • Shape [batch, features, anchors] - requires manual post-processing")

def save_analysis_report(info1, info2, filename="tflite_comparison_report.json"):
    """Save detailed comparison report to JSON."""
    report = {
        'model_1': info1,
        'model_2': info2,
        'summary': {
            'tensor_count_match': info1['tensor_count'] == info2['tensor_count'],
            'input_count_match': len(info1['input_details']) == len(info2['input_details']),
            'output_count_match': len(info1['output_details']) == len(info2['output_details']),
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\n✓ Detailed report saved to: {filename}")

def print_detailed_outputs(outputs1, outputs2, info1, info2):
    """Print detailed comparison of actual outputs."""
    print(f"\n{'#'*80}")
    print("DETAILED OUTPUT COMPARISON")
    print(f"{'#'*80}\n")
    
    print("=" * 80)
    print("MODEL 1 OUTPUTS")
    print("=" * 80)
    for i, (output, detail) in enumerate(zip(outputs1, info1['output_details'])):
        print(f"\n[Output {i}] {detail['name']}")
        print(f"  Shape: {output.shape}")
        print(f"  Values shape breakdown: {list(output.shape)}")
        
        if len(output.shape) == 3:  # [batch, features, anchors]
            print(f"\n  First anchor predictions (showing all {output.shape[1]} features):")
            for feat_idx in range(output.shape[1]):
                feat_name = ["x", "y", "w", "h", "obj", "class"][feat_idx] if feat_idx < 6 else f"feat_{feat_idx}"
                print(f"    {feat_name}: {output[0, feat_idx, 0]:.6f}")
            
            print(f"\n  Statistics across all {output.shape[2]} anchors:")
            for feat_idx in range(min(6, output.shape[1])):
                feat_name = ["x", "y", "w", "h", "objectness", "class_0"][feat_idx]
                values = output[0, feat_idx, :]
                print(f"    {feat_name}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        else:
            print(f"  Full output:\n{output}")
    
    print("\n" + "=" * 80)
    print("MODEL 2 OUTPUTS (Post-Processed)")
    print("=" * 80)
    for i, (output, detail) in enumerate(zip(outputs2, info2['output_details'])):
        print(f"\n[Output {i}] {detail['name']}")
        print(f"  Shape: {output.shape}")
        
        if i == 0:  # Bounding boxes
            print("  Bounding boxes [y_min, x_min, y_max, x_max] normalized:")
            for det_idx in range(output.shape[1]):
                box = output[0, det_idx]
                print(f"    Detection {det_idx}: [{box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}]")
        elif i == 1:  # Classes
            print("  Class IDs:")
            print(f"    {output[0]}")
        elif i == 2:  # Scores
            print("  Confidence scores:")
            for det_idx in range(output.shape[1]):
                score = output[0, det_idx]
                print(f"    Detection {det_idx}: {score:.4f}")
        elif i == 3:  # Number of detections
            print(f"  Number of valid detections: {int(output[0])}")
        else:
            print(f"  Full output:\n{output}")

def main(model1_path, model2_path, image_path=None):
    """Main analysis function."""
    print(f"\n{'#'*80}")
    print(f"TFLite Model Comparison Analysis")
    print(f"{'#'*80}")
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    if image_path:
        print(f"Test Image: {image_path}")
    
    # Analyze both models
    print("\n[1/5] Analyzing Model 1...")
    info1, interp1 = analyze_tflite_model(model1_path)
    
    print("[2/5] Analyzing Model 2...")
    info2, interp2 = analyze_tflite_model(model2_path)
    
    # Compare inputs
    print("\n[3/5] Comparing model structures...")
    compare_tensor_lists(info1['input_details'], info2['input_details'], "Inputs")
    
    # Compare outputs
    compare_tensor_lists(info1['output_details'], info2['output_details'], "Outputs")
    
    # Test inference
    print("\n[4/5] Testing inference on both models...")
    outputs1 = test_inference(interp1, "Model 1", image_path)
    outputs2 = test_inference(interp2, "Model 2", image_path)
    
    # Compare output formats
    print("\n[5/5] Comparing output formats...")
    compare_output_format(outputs1, outputs2)
    
    # Print detailed outputs
    print_detailed_outputs(outputs1, outputs2, info1, info2)
    
    # Print summary
    print_summary(info1, info2)
    
    # Save detailed report
    save_analysis_report(info1, info2)
    
    print(f"\n{'#'*80}")
    print("Analysis Complete!")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    # Replace these paths with your actual TFLite file paths
    MODEL1_PATH = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/detect.tflite'
    MODEL2_PATH = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/best_int8.tflite'
    IMAGE_PATH = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640_image.png'  # Set to an image path like "test_image.jpg" to test with real image
    
    main(MODEL1_PATH, MODEL2_PATH, IMAGE_PATH)

    