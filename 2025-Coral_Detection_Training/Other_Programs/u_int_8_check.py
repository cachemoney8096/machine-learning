import tensorflow as tf

def check_tflite_model_dtype(model_path):
    """
    Check the data type (quantization) of a TFLite model.
    """
    print(f"Analyzing model: {model_path}\n")
    print("="*70)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("INPUT TENSOR:")
        print("-" * 70)
        for i, input_detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    Name: {input_detail['name']}")
            print(f"    Shape: {input_detail['shape']}")
            print(f"    Dtype: {input_detail['dtype']}")
            print(f"    Dtype name: {input_detail['dtype'].__name__}")
            
            # Check for quantization parameters
            if 'quantization' in input_detail:
                quant = input_detail['quantization']
                print(f"    Quantization: {quant}")
            if 'quantization_parameters' in input_detail:
                quant_params = input_detail['quantization_parameters']
                print(f"    Quantization params: {quant_params}")
        
        print("\nOUTPUT TENSOR:")
        print("-" * 70)
        for i, output_detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    Name: {output_detail['name']}")
            print(f"    Shape: {output_detail['shape']}")
            print(f"    Dtype: {output_detail['dtype']}")
            print(f"    Dtype name: {output_detail['dtype'].__name__}")
            
            # Check for quantization parameters
            if 'quantization' in output_detail:
                quant = output_detail['quantization']
                print(f"    Quantization: {quant}")
            if 'quantization_parameters' in output_detail:
                quant_params = output_detail['quantization_parameters']
                print(f"    Quantization params: {quant_params}")
        
        print("\n" + "="*70)
        print("SUMMARY:")
        print("-" * 70)
        
        # Determine model type
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        
        if input_dtype == tf.uint8:
            print("✓ INPUT is UINT8 (quantized)")
        elif input_dtype == tf.float32:
            print("✓ INPUT is FLOAT32 (not quantized)")
        elif input_dtype == tf.float16:
            print("✓ INPUT is FLOAT16")
        else:
            print(f"✓ INPUT is {input_dtype.__name__}")
        
        if output_dtype == tf.uint8:
            print("✓ OUTPUT is UINT8 (quantized)")
        elif output_dtype == tf.float32:
            print("✓ OUTPUT is FLOAT32 (not quantized)")
        elif output_dtype == tf.float16:
            print("✓ OUTPUT is FLOAT16")
        else:
            print(f"✓ OUTPUT is {output_dtype.__name__}")
        
        print("="*70)
        
        # Return the types for programmatic use
        return {
            'input_dtype': input_dtype,
            'output_dtype': output_dtype,
            'is_input_uint8': input_dtype == tf.uint8,
            'is_output_uint8': output_dtype == tf.uint8,
            'is_fully_quantized': input_dtype == tf.uint8 and output_dtype == tf.uint8
        }
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Check your int8 model
    model_path = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/output_transpose_only.tflite'
    # model_path = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/ssdlite_object_detection.tflite'
    
    result = check_tflite_model_dtype(model_path)
    
    if result:
        print("\nQuick check:")
        if result['is_fully_quantized']:
            print("→ This is a FULLY QUANTIZED (int8/uint8) model")
        elif result['is_input_uint8'] or result['is_output_uint8']:
            print("→ This is a PARTIALLY QUANTIZED model")
        else:
            print("→ This is a FLOATING POINT model")