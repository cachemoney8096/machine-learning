import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best_saved_model/best_float32.tflite"
)
interpreter.allocate_tensors()

for o in interpreter.get_output_details():
    print(o["shape"], o["dtype"])
