import tensorflow as tf
import sys

def get_tflite_dimensions(tflite_path):
    """
    Get input and output dimensions of a TFLite model.
    """
    print("=" * 80)
    print("TFLite Model Dimensions")
    print("=" * 80)
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📁 Model: {tflite_path}")
    print(f"📊 Number of inputs: {len(input_details)}")
    print(f"📊 Number of outputs: {len(output_details)}")
    
    # Print input dimensions
    print("\n" + "-" * 80)
    print("INPUT TENSORS")
    print("-" * 80)
    for i, detail in enumerate(input_details):
        print(f"\nInput {i}:")
        print(f"  Name:  {detail['name']}")
        print(f"  Shape: {detail['shape'].tolist()}")
        print(f"  Type:  {detail['dtype'].__name__}")
        if 'quantization_parameters' in detail:
            quant = detail['quantization_parameters']
            if quant['scales'].size > 0:
                print(f"  Quantization: scales={quant['scales']}, zero_points={quant['zero_points']}")
    
    # Print output dimensions
    print("\n" + "-" * 80)
    print("OUTPUT TENSORS")
    print("-" * 80)
    for i, detail in enumerate(output_details):
        print(f"\nOutput {i}:")
        print(f"  Name:  {detail['name']}")
        print(f"  Shape: {detail['shape'].tolist()}")
        print(f"  Type:  {detail['dtype'].__name__}")
        if 'quantization_parameters' in detail:
            quant = detail['quantization_parameters']
            if quant['scales'].size > 0:
                print(f"  Quantization: scales={quant['scales']}, zero_points={quant['zero_points']}")
    
    print("\n" + "=" * 80)
    
    return input_details, output_details


def quick_check(tflite_path):
    """Quick one-line check of dimensions."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_shape = interpreter.get_input_details()[0]['shape']
    output_shape = interpreter.get_output_details()[0]['shape']
    
    print(f"Input:  {input_shape.tolist()}")
    print(f"Output: {output_shape.tolist()}")
    
    return input_shape, output_shape


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <model.tflite>")
    #     print("\nOr enter path interactively:")
    #     tflite_path = input("📁 TFLite model path: ").strip()
    # else:
    #     tflite_path = sys.argv[1]
    tflite_path = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/output_transpose_only.tflite'
    try:
        get_tflite_dimensions(tflite_path)
    except Exception as e:
        print(f"❌ Error: {e}")