import cv2
import numpy as np
import tensorflow as tf

def debug_tflite_model(model_path, video_path):
    """
    Debug a TFLite model using a video file.
    This will help identify why you're getting no predictions.
    """
    print("="*60)
    print("TFLITE MODEL DEBUGGER")
    print("="*60)
    
    # Load the model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("INPUT DETAILS:")
    print("-" * 60)
    for i, detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Quantization: {detail.get('quantization', 'None')}")
    
    print("\nOUTPUT DETAILS:")
    print("-" * 60)
    for i, detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Type: {detail['dtype']}")
        print(f"  Quantization: {detail.get('quantization', 'None')}")
    
    # Load and preprocess test frame from video
    print("\n" + "="*60)
    print("LOADING VIDEO")
    print("="*60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video loaded: {vid_width}x{vid_height} @ {fps}fps, {total_frames} frames")
    
    # Read first frame
    ret, img = cap.read()
    cap.release()
    
    if not ret:
        print("✗ Could not read frame from video")
        return
    
    print(f"✓ Read first frame: {img.shape}")
    
    # Get input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    
    # Preprocess
    input_data = cv2.resize(img, (width, height))
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    
    print(f"✓ Resized to: {input_data.shape}")
    
    print("\n" + "="*60)
    print("RUNNING TEST INFERENCE")
    print("="*60)
    
    # Show preprocessing options
    input_data_expanded = np.expand_dims(input_data, axis=0)
    
    # Try different preprocessing approaches
    if input_details[0]['dtype'] == np.float32:
        print("\nInput type is FLOAT32 - trying normalized input (0-1)")
        input_data_final = input_data_expanded.astype(np.float32) / 255.0
    elif input_details[0]['dtype'] == np.uint8:
        print("\nInput type is UINT8 - using unnormalized input (0-255)")
        input_data_final = input_data_expanded.astype(np.uint8)
    else:
        print(f"\nUnexpected input type: {input_details[0]['dtype']}")
        input_data_final = input_data_expanded
    
    print(f"Input data shape: {input_data_final.shape}")
    print(f"Input data type: {input_data_final.dtype}")
    print(f"Input data range: [{input_data_final.min():.3f}, {input_data_final.max():.3f}]")
    
    # Run inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data_final)
        interpreter.invoke()
        print("\n✓ Inference completed successfully")
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        return
    
    # Get outputs
    print("\n" + "="*60)
    print("OUTPUT ANALYSIS")
    print("="*60)
    
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        print(f"\nOutput {i} ({detail['name']}):")
        print(f"  Shape: {output.shape}")
        print(f"  Type: {output.dtype}")
        print(f"  Range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  Mean: {output.mean():.6f}")
        print(f"  Non-zero elements: {np.count_nonzero(output)}/{output.size}")
        
        # Show first few values for inspection
        if output.size <= 20:
            print(f"  Values: {output.flatten()}")
        else:
            print(f"  First 20 values: {output.flatten()[:20]}")
    
    # Analyze detection format
    print("\n" + "="*60)
    print("DETECTION FORMAT ANALYSIS")
    print("="*60)
    
    if len(output_details) == 4:
        print("\nDetected format: STANDARD (boxes, classes, scores, num_detections)")
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
        
        print(f"Number of detections: {num_detections}")
        print(f"\nTop 5 scores: {sorted(scores, reverse=True)[:5]}")
        
        for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
            count = np.sum(scores >= threshold)
            print(f"Detections above {threshold:.2f}: {count}")
        
    elif len(output_details) == 1:
        print("\nDetected format: YOLO-style (single output)")
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"Output shape: {output.shape}")
        
        if len(output.shape) == 2 and output.shape[1] >= 6:
            print(f"Appears to be YOLO format: [{output.shape[0]} detections, {output.shape[1]} values per detection]")
            print(f"Format likely: [x_center, y_center, w, h, confidence, class_scores...]")
            
            # Analyze confidence scores
            confidences = output[:, 4]
            print(f"\nConfidence statistics:")
            print(f"  Range: [{confidences.min():.6f}, {confidences.max():.6f}]")
            print(f"  Mean: {confidences.mean():.6f}")
            print(f"  Median: {np.median(confidences):.6f}")
            
            for threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
                count = np.sum(confidences >= threshold)
                print(f"  Above {threshold:.2f}: {count}")
            
            # Show top detections
            print("\nTop 5 detections:")
            top_indices = np.argsort(confidences)[-5:][::-1]
            for idx in top_indices:
                det = output[idx]
                if len(det) >= 6:
                    x, y, w, h, conf = det[:5]
                    class_scores = det[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    print(f"  Box: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}), "
                          f"Conf: {conf:.4f}, Class {class_id}: {class_conf:.4f}")
        else:
            print(f"Unexpected output shape: {output.shape}")
            print("Cannot automatically parse this format")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Provide recommendations
    max_score = 0
    if len(output_details) == 4:
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        max_score = scores.max()
    elif len(output_details) == 1:
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        if len(output.shape) == 2 and output.shape[1] >= 5:
            max_score = output[:, 4].max()
    
    if max_score < 0.01:
        print("⚠ Maximum confidence is very low (< 0.01)")
        print("  Possible issues:")
        print("  1. Wrong input preprocessing (check normalization)")
        print("  2. Model expects different input format (try 0-255 vs 0-1)")
        print("  3. Model wasn't trained properly")
        print("  4. Test image doesn't contain objects the model was trained on")
    elif max_score < 0.1:
        print("⚠ Maximum confidence is low (< 0.1)")
        print("  Try lowering confidence_threshold to 0.01 or 0.05")
    else:
        print(f"✓ Found detections with confidence up to {max_score:.3f}")
        print("  Your code should be working - check threshold settings")


if __name__ == "__main__":
    # REPLACE THESE PATHS WITH YOUR ACTUAL FILES
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best_float32.tflite"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/people_walking_640x640.mp4"
    
    debug_tflite_model(model_path, video_path)