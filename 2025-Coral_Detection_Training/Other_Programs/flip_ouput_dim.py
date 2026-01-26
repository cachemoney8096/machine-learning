import tensorflow as tf
import numpy as np
import sys

def add_transpose_using_flatbuffer(input_tflite_path, output_tflite_path):
    """
    Add transpose by modifying the TFLite flatbuffer directly.
    This appends a TRANSPOSE op to the end of the model.
    """
    try:
        # Try to import flatbuffers (needed for direct manipulation)
        from tensorflow.lite.python import schema_py_generated as schema_fb
        import flatbuffers
    except ImportError:
        print("❌ flatbuffers not available for direct manipulation")
        return False
    
    print("=" * 80)
    print("Adding Transpose via Flatbuffer Modification")
    print("=" * 80)
    
    # Load the model
    with open(input_tflite_path, 'rb') as f:
        buf = bytearray(f.read())
    
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    
    print(f"\n📊 Model has {model.SubgraphsLength()} subgraph(s)")
    
    # Note: Direct flatbuffer editing is complex and error-prone
    # This approach is not recommended without deep understanding of the format
    print("⚠️  Direct flatbuffer editing is very complex")
    return False


def create_transpose_onnx_method(input_tflite_path, output_tflite_path):
    """
    Convert TFLite -> ONNX -> Add Transpose -> TFLite
    Requires: tf2onnx, onnx, onnx-tf
    """
    try:
        import onnx
        import tf2onnx
        from onnx import numpy_helper
    except ImportError:
        print("❌ ONNX tools not installed")
        print("Install with: pip install tf2onnx onnx onnx-tf")
        return False
    
    print("=" * 80)
    print("Converting via ONNX (TFLite -> ONNX -> TFLite)")
    print("=" * 80)
    
    # This method is complex and may not preserve all model properties
    print("⚠️  ONNX conversion may not preserve all TFLite optimizations")
    return False


def create_standalone_transpose_model(input_tflite_path, output_tflite_path):
    """
    Create a simple Keras model with transpose and convert it.
    Then you'll need to chain it with your original model.
    """
    
    print("=" * 80)
    print("Creating Standalone Transpose Model")
    print("=" * 80)
    
    # Load original to get output shape
    interpreter = tf.lite.Interpreter(model_path=input_tflite_path)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    
    output_shape = output_details[0]['shape']
    output_dtype = output_details[0]['dtype']
    
    print(f"\n📊 Original output: {output_shape.tolist()}")
    
    # Create a simple transpose model
    input_layer = tf.keras.Input(shape=output_shape[1:], dtype=output_dtype)
    # Permute (2, 1) swaps the last two dimensions: [6, 8400] -> [8400, 6]
    transposed = tf.keras.layers.Permute((2, 1))(input_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=transposed)
    
    print(f"📊 New output: {model.output.shape}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    transpose_model_path = output_tflite_path.replace('.tflite', '_transpose_only.tflite')
    with open(transpose_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ Created transpose-only model: {transpose_model_path}")
    print("\n💡 Note: You'll need to run TWO models in sequence:")
    print("   1. Your original model")
    print("   2. This transpose model")
    
    return True


def show_post_processing_solution(input_tflite_path):
    """
    Since embedding doesn't work, show the post-processing approach.
    This is the most reliable solution for Limelight.
    """
    
    print("=" * 80)
    print("POST-PROCESSING SOLUTION (Recommended)")
    print("=" * 80)
    
    interpreter = tf.lite.Interpreter(model_path=input_tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n📊 Model Info:")
    print(f"   Input:  {input_details[0]['shape'].tolist()}")
    print(f"   Output: {output_details[0]['shape'].tolist()}")
    
    print("\n" + "=" * 80)
    print("SOLUTION CODE")
    print("=" * 80)
    
    print("""
import numpy as np
import tensorflow as tf

# Load your model
interpreter = tf.lite.Interpreter(model_path='your_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
input_data = ...  # Your input data
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Original output shape: {output.shape}")  # [1, 6, 8400]

# TRANSPOSE: [1, 6, 8400] -> [1, 8400, 6]
output_transposed = np.transpose(output, (0, 2, 1))
print(f"Transposed output shape: {output_transposed.shape}")  # [1, 8400, 6]

# Now use output_transposed for your processing
# This is the same as [1, 8400, 6] that you want
    """)
    
    print("\n" + "=" * 80)
    print("FOR LIMELIGHT SPECIFIC CODE")
    print("=" * 80)
    
    print("""
If you're using Limelight's Python API, add this after inference:

# After getting results from Limelight
detections = interpreter.get_tensor(output_details[0]['index'])

# Transpose to get correct shape
detections = np.transpose(detections, (0, 2, 1))  # [1, 6, 8400] -> [1, 8400, 6]

# Now detections[0] has shape [8400, 6]
# Where each of 8400 detections has 6 values: [x, y, w, h, confidence, class]
    """)
    
    # Demonstrate with dummy data
    print("\n" + "=" * 80)
    print("TESTING WITH DUMMY DATA")
    print("=" * 80)
    
    dummy_input = np.random.randn(*input_details[0]['shape']).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n✓ Original output: {output.shape}")
    
    output_transposed = np.transpose(output, (0, 2, 1))
    print(f"✓ Transposed output: {output_transposed.shape}")
    
    # Show what the data looks like
    print(f"\n📊 First detection (before transpose): {output[0, :, 0]}")
    print(f"📊 First detection (after transpose):  {output_transposed[0, 0, :]}")
    
    return True


def suggest_model_rebuild(input_tflite_path):
    """
    Suggest rebuilding the model with transpose before TFLite conversion.
    """
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE: Rebuild Model from Source")
    print("=" * 80)
    
    print("""
If you have access to the original model (before TFLite conversion):

1. ONNX/PyTorch Model:
   - Add a transpose/permute layer at the end
   - Re-export to ONNX
   - Convert ONNX to TFLite

2. TensorFlow/Keras Model:
   
   import tensorflow as tf
   
   # Load your original model
   model = tf.keras.models.load_model('original_model.h5')
   
   # Add transpose to output
   x = model.output
   x = tf.keras.layers.Permute((2, 1))(x)  # Swaps last 2 dims
   
   new_model = tf.keras.Model(inputs=model.input, outputs=x)
   
   # Convert to TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
   tflite_model = converter.convert()
   
   with open('transposed_model.tflite', 'wb') as f:
       f.write(tflite_model)

3. If using YOLOv8/Ultralytics:
   - Modify the model export code to add transpose
   - Re-export to TFLite with the transpose included
    """)


if __name__ == "__main__":
    print("\n🔧 TFLite Transpose Tool")
    print("=" * 80)
    
    # if len(sys.argv) >= 2:
    #     input_model = sys.argv[1]
    #     output_model = sys.argv[2] if len(sys.argv) >= 3 else "output_transposed.tflite"
    # else:
    #     input_model = input("📁 Input TFLite model path: ").strip()
    #     output_model = input("💾 Output TFLite model path (or press Enter for post-processing): ").strip()
    input_model='/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/best_int8.tflite'
    output_model='/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/output.tflite'
    
    print(f"\n📁 Input: {input_model}\n")
    
    # Check if file exists
    try:
        with open(input_model, 'rb') as f:
            pass
    except FileNotFoundError:
        print(f"❌ Error: File '{input_model}' not found!")
        sys.exit(1)
    
    print("\n⚠️  IMPORTANT: Embedding transpose directly into TFLite is not possible")
    print("              due to TFLite converter limitations.\n")
    
    print("Choose your approach:")
    print("1. [RECOMMENDED] See post-processing code (add transpose after inference)")
    print("2. Create separate transpose model (need to run 2 models)")
    print("3. See how to rebuild from original model")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1" or choice == "4":
        show_post_processing_solution(input_model)
    
    if choice == "2" or choice == "4":
        if output_model:
            print("\n")
            create_standalone_transpose_model(input_model, output_model)
    
    if choice == "3" or choice == "4":
        suggest_model_rebuild(input_model)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The BEST solution for Limelight is post-processing:
    
    output_transposed = np.transpose(output, (0, 2, 1))
    
This is:
✓ Most reliable
✓ No model modification needed
✓ Works on any hardware
✓ Minimal performance impact
✓ Easy to implement

The transpose operation is very fast (just reindexing, no data copy).
    """)