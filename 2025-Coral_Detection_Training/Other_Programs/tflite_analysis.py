import tensorflow as tf
import numpy as np
from collections import defaultdict

def analyze_quantization(model_path):
    """Comprehensive quantization analysis of a TFLite model."""
    
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    print(f"\n{'='*80}")
    print(f"QUANTIZATION ANALYSIS: {model_path}")
    print(f"{'='*80}\n")
    
    # Analyze quantization scheme
    quant_types = defaultdict(int)
    quantized_tensors = []
    float_tensors = []
    
    for tensor in tensor_details:
        dtype = tensor['dtype']
        quant = tensor['quantization']
        quant_params = tensor.get('quantization_parameters', {})
        
        quant_types[str(dtype)] += 1
        
        # Check if quantized (scale != 0)
        if quant[0] != 0 or (quant_params and quant_params.get('scales') is not None):
            quantized_tensors.append(tensor)
        else:
            float_tensors.append(tensor)
    
    # Print overall statistics
    print("📊 OVERALL QUANTIZATION SUMMARY")
    print("-" * 80)
    print(f"Total tensors: {len(tensor_details)}")
    print(f"Quantized tensors: {len(quantized_tensors)}")
    print(f"Float tensors: {len(float_tensors)}")
    print(f"\nData type distribution:")
    for dtype, count in sorted(quant_types.items()):
        print(f"  {dtype}: {count} tensors")
    
    # Determine quantization type
    print(f"\n🔍 QUANTIZATION TYPE DETECTION")
    print("-" * 80)
    
    has_int8 = any(str(t['dtype']) == "<class 'numpy.int8'>" for t in tensor_details)
    has_uint8 = any(str(t['dtype']) == "<class 'numpy.uint8'>" for t in tensor_details)
    has_int16 = any(str(t['dtype']) == "<class 'numpy.int16'>" for t in tensor_details)
    has_float16 = any(str(t['dtype']) == "<class 'numpy.float16'>" for t in tensor_details)
    has_float32 = any(str(t['dtype']) == "<class 'numpy.float32'>" for t in tensor_details)
    
    if has_int8 or has_uint8:
        if len(quantized_tensors) == len(tensor_details):
            print("✓ FULL INTEGER QUANTIZATION (INT8/UINT8)")
            print("  All tensors are quantized - maximum efficiency")
        elif input_details[0]['dtype'] in [np.float32, np.float16]:
            print("✓ DYNAMIC RANGE QUANTIZATION")
            print("  Weights quantized, but inputs/outputs are float")
        else:
            print("✓ INTEGER QUANTIZATION with FLOAT FALLBACK")
            print("  Most operations quantized, some ops use float")
    elif has_float16:
        print("✓ FLOAT16 QUANTIZATION")
        print("  Model uses 16-bit floating point")
    elif has_float32 and len(quantized_tensors) == 0:
        print("✗ NO QUANTIZATION (Full Float32)")
        print("  Model uses full 32-bit floating point precision")
    else:
        print("⚠️  MIXED/CUSTOM QUANTIZATION")
        print("  Model uses a combination of data types")
    
    # Input/Output quantization
    print(f"\n📥 INPUT QUANTIZATION")
    print("-" * 80)
    for i, inp in enumerate(input_details):
        print(f"\nInput {i}: {inp['name']}")
        print(f"  Data type: {inp['dtype']}")
        print(f"  Shape: {inp['shape']}")
        print(f"  Quantization: {inp['quantization']}")
        
        if inp['quantization'][0] != 0:
            scale, zero_point = inp['quantization']
            print(f"  ✓ QUANTIZED:")
            print(f"    Scale: {scale}")
            print(f"    Zero point: {zero_point}")
            print(f"    Formula: real_value = scale * (quantized_value - zero_point)")
            print(f"    Range: [{scale * (0 - zero_point):.4f}, {scale * (255 - zero_point):.4f}]")
        else:
            print(f"  ✗ NOT QUANTIZED (Float)")
    
    print(f"\n📤 OUTPUT QUANTIZATION")
    print("-" * 80)
    for i, out in enumerate(output_details):
        print(f"\nOutput {i}: {out['name']}")
        print(f"  Data type: {out['dtype']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Quantization: {out['quantization']}")
        
        if out['quantization'][0] != 0:
            scale, zero_point = out['quantization']
            print(f"  ✓ QUANTIZED:")
            print(f"    Scale: {scale}")
            print(f"    Zero point: {zero_point}")
            print(f"    Formula: real_value = scale * (quantized_value - zero_point)")
        else:
            print(f"  ✗ NOT QUANTIZED (Float)")
    
    # Detailed tensor analysis
    if quantized_tensors:
        print(f"\n🔢 QUANTIZED TENSORS DETAILS")
        print("-" * 80)
        print(f"Showing first 10 of {len(quantized_tensors)} quantized tensors:\n")
        
        for i, tensor in enumerate(quantized_tensors[:10]):
            scale, zero_point = tensor['quantization']
            print(f"{i+1}. {tensor['name']}")
            print(f"   Type: {tensor['dtype']}, Shape: {tensor['shape']}")
            print(f"   Scale: {scale:.8f}, Zero point: {zero_point}")
            
            # Check for per-channel quantization
            quant_params = tensor.get('quantization_parameters', {})
            if quant_params.get('scales') is not None:
                scales = quant_params['scales']
                if len(scales) > 1:
                    print(f"   ⚠️  PER-CHANNEL QUANTIZATION: {len(scales)} channels")
                    print(f"      Scale range: [{min(scales):.8f}, {max(scales):.8f}]")
            print()
    
    # Quantization recommendations
    print(f"\n💡 ANALYSIS & RECOMMENDATIONS")
    print("-" * 80)
    
    if has_float32 and len(quantized_tensors) == 0:
        print("⚠️  This model is NOT quantized")
        print("   Consider quantizing for:")
        print("   • Faster inference (2-4x speedup)")
        print("   • Smaller model size (4x reduction)")
        print("   • Lower power consumption")
        print("   • Better deployment on mobile/edge devices")
    
    elif input_details[0]['dtype'] == np.uint8 and output_details[0]['dtype'] in [np.float32, np.float16]:
        print("✓ Model uses quantized input but float output")
        print("   This is common for object detection models")
        print("   Input preprocessing is hardware-accelerated")
    
    elif input_details[0]['dtype'] in [np.float32, np.float16]:
        print("⚠️  Model has float input despite quantized weights")
        print("   Consider using integer input for:")
        print("   • Better hardware acceleration")
        print("   • Faster preprocessing")
        print("   • End-to-end quantization benefits")
    
    if len(quantized_tensors) > 0 and len(float_tensors) > 0:
        quant_ratio = len(quantized_tensors) / len(tensor_details) * 100
        print(f"\nℹ️  Quantization coverage: {quant_ratio:.1f}%")
        if quant_ratio < 80:
            print("   Consider full quantization for maximum efficiency")
    
    # Model size estimation
    print(f"\n📦 MODEL SIZE ESTIMATION")
    print("-" * 80)
    total_params = 0
    for tensor in tensor_details:
        if tensor['shape']:
            params = np.prod(tensor['shape'])
            total_params += params
    
    # Estimate bytes based on data types
    bytes_per_param = {
        "<class 'numpy.float32'>": 4,
        "<class 'numpy.float16'>": 2,
        "<class 'numpy.int8'>": 1,
        "<class 'numpy.uint8'>": 1,
        "<class 'numpy.int16'>": 2,
    }
    
    estimated_size = 0
    for dtype, count in quant_types.items():
        # Rough estimation
        estimated_size += count * bytes_per_param.get(dtype, 4) * 1000  # Rough multiplier
    
    print(f"Estimated parameters: ~{total_params:,}")
    print(f"Approximate model size: ~{estimated_size / (1024*1024):.2f} MB")
    
    import os
    if os.path.exists(model_path):
        actual_size = os.path.getsize(model_path)
        print(f"Actual file size: {actual_size / (1024*1024):.2f} MB")

def compare_quantization(model1_path, model2_path):
    """Compare quantization between two models."""
    print(f"\n{'#'*80}")
    print("COMPARING QUANTIZATION BETWEEN TWO MODELS")
    print(f"{'#'*80}\n")
    
    analyze_quantization(model1_path)
    print("\n" + "="*80 + "\n")
    analyze_quantization(model2_path)

if __name__ == "__main__":
    # Single model analysis
    MODEL_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites//Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/limelight_ready_6.tflite"
    analyze_quantization(MODEL_PATH)
    
    # Or compare two models
    # MODEL1_PATH = "model1.tflite"
    # MODEL2_PATH = "model2.tflite"
    # compare_quantization(MODEL1_PATH, MODEL2_PATH)