import cv2
import numpy as np
import tensorflow as tf


def diagnose_tflite_model(model_path, test_image_path):
    """
    Diagnose TFLite model output to understand its format.
    """
    print("="*70)
    print("TFLite Model Diagnostic Tool")
    print("="*70)
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n--- MODEL INFO ---")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Number of outputs: {len(output_details)}")
    
    for i, out in enumerate(output_details):
        print(f"\nOutput {i}:")
        print(f"  Name: {out['name']}")
        print(f"  Shape: {out['shape']}")
        print(f"  Dtype: {out['dtype']}")
    
    # Load and preprocess test image
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"\nError: Could not load image from {test_image_path}")
        return
    
    print(f"\n--- IMAGE INFO ---")
    print(f"Original image shape: {img.shape}")
    
    # Simple resize (first test)
    img_resized = cv2.resize(img, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    if input_details[0]['dtype'] == np.float32:
        img_input = img_rgb.astype(np.float32) / 255.0
    else:
        img_input = img_rgb.astype(np.uint8)
    
    img_input = np.expand_dims(img_input, axis=0)
    
    print(f"Preprocessed input shape: {img_input.shape}")
    print(f"Input value range: [{img_input.min():.4f}, {img_input.max():.4f}]")
    
    # Run inference
    print("\n--- RUNNING INFERENCE ---")
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Get all outputs
    print("\n--- RAW OUTPUT ANALYSIS ---")
    outputs = []
    for i, out_detail in enumerate(output_details):
        output = interpreter.get_tensor(out_detail['index'])
        outputs.append(output)
        print(f"\nOutput {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min: {output.min():.6f}")
        print(f"  Max: {output.max():.6f}")
        print(f"  Mean: {output.mean():.6f}")
        print(f"  Std: {output.std():.6f}")
        
        # Show first few values
        flat = output.flatten()
        print(f"  First 10 values: {flat[:10]}")
        print(f"  Last 10 values: {flat[-10:]}")
    
    # Analyze the main output (usually first one)
    main_output = outputs[0]
    
    print("\n--- DETAILED ANALYSIS OF MAIN OUTPUT ---")
    
    # Remove batch dimension if present
    if len(main_output.shape) == 3:
        main_output = main_output[0]
        print(f"After removing batch dim: {main_output.shape}")
    
    # Check if we need to transpose
    print(f"\nShape analysis:")
    print(f"  Dimension 0: {main_output.shape[0]}")
    print(f"  Dimension 1: {main_output.shape[1]}")
    
    # Try both orientations
    for orientation_name, oriented_output in [("Original", main_output), ("Transposed", main_output.T)]:
        print(f"\n--- Testing {orientation_name} orientation ---")
        print(f"Shape: {oriented_output.shape}")
        
        if len(oriented_output.shape) != 2:
            continue
        
        num_boxes, num_values = oriented_output.shape
        print(f"Interpreting as: {num_boxes} boxes × {num_values} values")
        
        if num_values < 5:
            print("  ❌ Too few values per box (need at least 5)")
            continue
        
        print(f"  ✓ Could be valid format")
        
        # Analyze first few boxes
        print(f"\n  First 5 boxes:")
        for i in range(min(5, len(oriented_output))):
            box = oriented_output[i]
            print(f"    Box {i}: {box[:min(10, len(box))]}")
            
            # Try to interpret as [x, y, w, h, conf, ...]
            if len(box) >= 5:
                x, y, w, h, conf = box[0], box[1], box[2], box[3], box[4]
                print(f"      → x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}, conf={conf:.3f}")
                
                # Check if values make sense
                if 0 <= x <= input_width and 0 <= y <= input_height:
                    print(f"      ✓ Coordinates look like pixel values")
                elif 0 <= x <= 1 and 0 <= y <= 1:
                    print(f"      ✓ Coordinates look normalized (0-1)")
                else:
                    print(f"      ⚠ Coordinates outside expected range")
                
                if 0 <= conf <= 1:
                    print(f"      ✓ Confidence looks valid (0-1)")
                else:
                    print(f"      ⚠ Confidence outside 0-1 range")
        
        # Find max confidence
        if num_values >= 5:
            confidences = oriented_output[:, 4]
            max_conf = confidences.max()
            max_idx = confidences.argmax()
            print(f"\n  Max confidence: {max_conf:.6f} at box {max_idx}")
            print(f"  That box: {oriented_output[max_idx, :min(10, num_values)]}")
            
            # Count boxes above various thresholds
            for thresh in [0.01, 0.05, 0.1, 0.25, 0.5]:
                count = (confidences > thresh).sum()
                print(f"  Boxes with conf > {thresh}: {count}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    max_val = outputs[0].max()
    if max_val > 100:
        print("⚠ Output values are very large - may need different scaling")
    elif max_val < 0.01:
        print("⚠ Output values are very small - may need different scaling")
    elif max_val <= 1.0:
        print("✓ Output values in reasonable range (0-1)")
    
    print("\nNext steps:")
    print("1. Compare this output with your PyTorch model's output")
    print("2. Check if coordinates need different interpretation")
    print("3. Try exporting with different options (see below)")
    print("\n" + "="*70)


def compare_pt_vs_tflite(pt_model_path, tflite_model_path, test_image_path):
    """
    Compare PyTorch and TFLite model outputs side by side.
    """
    from ultralytics import YOLO
    
    print("\n" + "="*70)
    print("COMPARING PYTORCH VS TFLITE")
    print("="*70)
    
    # Load PyTorch model
    print("\n--- PYTORCH MODEL ---")
    pt_model = YOLO(pt_model_path)
    pt_results = pt_model.predict(test_image_path, verbose=False)[0]
    
    print(f"Number of detections: {len(pt_results.boxes)}")
    if len(pt_results.boxes) > 0:
        print(f"Confidences: {pt_results.boxes.conf.cpu().numpy()}")
        print(f"Boxes (xyxy): {pt_results.boxes.xyxy.cpu().numpy()}")
    
    # Now run TFLite diagnostic
    print("\n--- TFLITE MODEL ---")
    diagnose_tflite_model(tflite_model_path, test_image_path)


if __name__ == "__main__":
    # Test with single image first
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v16/weights/best_saved_model/best_float32.tflite"
    
    # Extract a frame from your video or use a test image
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Extract first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        test_image = "test_frame.jpg"
        cv2.imwrite(test_image, frame)
        print(f"Extracted test frame to {test_image}")
        
        # Run diagnostic
        diagnose_tflite_model(model_path, test_image)
        
        # Optional: Compare with PyTorch if you have the .pt file
        pt_model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v16/weights/best.pt"
        try:
            compare_pt_vs_tflite(pt_model_path, model_path, test_image)
        except Exception as e:
            print(f"\nCouldn't compare with PyTorch: {e}")
    else:
        print("Couldn't extract frame from video")