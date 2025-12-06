import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf

def run_detection_on_video(model_path, video_path, output_path=None, confidence_threshold=0.1, input_size=640):
    """
    Run object detection on every frame of a video using a .tflite model file.
    
    Args:
        model_path: Path to .tflite model file
        video_path: Path to input video file
        output_path: Path for output video (optional)
        confidence_threshold: Minimum confidence for detections (0-1)
        input_size: Input size for the model (default: 640)
    """
    # Load the TFLite model
    print(f"Loading TFLite model from {model_path}...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("Model loaded successfully!")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input type: {input_details[0]['dtype']}")
        print(f"Number of outputs: {len(output_details)}")

        # === ADDED: PRINT OUTPUT LAYER DETAILS ===
        print("\n=== OUTPUT LAYERS ===")
        for i, out in enumerate(output_details):
            print(f"Output #{i}")
            print(f"  Name:  {out['name']}")
            print(f"  Index: {out['index']}")
            print(f"  Shape: {out['shape']}")
            print(f"  Dtype: {out['dtype']}")
            print("----------------------------------")
        print("==================================\n")
        # =========================================
        
        # Get input dimensions
        input_shape = input_details[0]['shape']
        height = input_shape[1]
        width = input_shape[2]
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info: {vid_width}x{vid_height} @ {fps}fps, {total_frames} frames")
    
    # Set output path if not provided
    if output_path is None:
        input_file = Path(video_path)
        output_path = input_file.parent / f"{input_file.stem}_detected{input_file.suffix}"
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_width, vid_height))
    
    frame_count = 0
    total_detections = 0
    
    print("\nProcessing video frames...")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess frame
            input_data = cv2.resize(frame, (width, height))
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_data, axis=0)
            
            # Normalize based on input type
            if input_details[0]['dtype'] == np.float32:
                input_data = input_data.astype(np.float32) / 255.0
            else:
                input_data = input_data.astype(np.uint8)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get detection results
            # Different TFLite models have different output formats
            # Common formats: [boxes, classes, scores, num_detections] or [output_0]
            
            if len(output_details) == 4:
                # Format: boxes, classes, scores, num_detections
                boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
                classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
                scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
                num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
            elif len(output_details) == 1:
                # YOLO format: single output
                output = interpreter.get_tensor(output_details[0]['index'])
                
                # Debug: print shape on first frame
                if frame_count == 0:
                    print(f"\nDEBUG - Raw output shape: {output.shape}")
                
                # Remove batch dimension if present
                if len(output.shape) == 3:
                    output = output[0]
                    if frame_count == 0:
                        print(f"DEBUG - After removing batch: {output.shape}")
                
                # Check if output needs to be transposed
                # YOLO output can be (num_values, num_boxes) or (num_boxes, num_values)
                # We need (num_boxes, num_values) where num_values is typically 5-6
                if output.shape[0] < output.shape[1]:
                    output = output.T  # Transpose to (num_boxes, num_values)
                    if frame_count == 0:
                        print(f"DEBUG - After transpose: {output.shape}")
                
                # Parse YOLO output
                boxes = []
                classes = []
                scores = []
                
                detection_count = 0
                for detection in output:
                    # Debug: print first detection fully on first frame
                    if frame_count == 0 and detection_count == 0:
                        print(f"DEBUG - Full first detection: {detection}")
                        print(f"DEBUG - Detection length: {len(detection)}")
                    
                    if len(detection) >= 5:
                        # Try to find where confidence actually is
                        # Standard YOLO: [x, y, w, h, confidence, class_scores...]
                        # But some models use: [x, y, w, h, class_scores...] with no separate confidence
                        
                        x_center, y_center, w, h = detection[0:4]
                        
                        if len(detection) == 5:
                            # Single class: [x, y, w, h, confidence]
                            confidence = detection[4]
                            class_id = 0
                            class_conf = 1.0
                        elif len(detection) == 6:
                            # Could be [x, y, w, h, conf, class] OR [x, y, w, h, class1, class2]
                            # Check if index 4 looks like confidence (0-1) or class score
                            confidence = detection[4]
                            class_scores = detection[5:]
                            class_id = np.argmax(class_scores)
                            class_conf = class_scores[class_id]
                        else:
                            # Multi-class without separate confidence: [x, y, w, h, class_scores...]
                            # The confidence IS the max class score
                            class_scores = detection[4:]
                            class_id = np.argmax(class_scores)
                            confidence = class_scores[class_id]
                            class_conf = 1.0
                        
                        # Debug: print top confidences on first frame
                        if frame_count == 0 and detection_count < 5:
                            print(f"DEBUG - Detection {detection_count}: conf={confidence:.4f}, box=({x_center:.1f}, {y_center:.1f}, {w:.1f}, {h:.1f})")
                            if len(detection) > 5:
                                print(f"  Class scores: {detection[4:].tolist()}")
                            detection_count += 1
                        
                        if confidence >= confidence_threshold:
                            
                            # YOLO outputs coordinates relative to input size
                            # Convert to normalized [0-1] coordinates
                            x1 = (x_center - w/2) / width
                            y1 = (y_center - h/2) / height
                            x2 = (x_center + w/2) / width
                            y2 = (y_center + h/2) / height
                            
                            # Clamp to valid range
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(1, x2), min(1, y2)
                            
                            boxes.append([x1, y1, x2, y2])
                            classes.append(class_id)
                            scores.append(float(confidence))
                
                if frame_count == 0:
                    print(f"DEBUG - Found {len(boxes)} detections above threshold {confidence_threshold}")
                
                # Apply NMS to remove duplicate detections
                if len(boxes) > 0:
                    boxes_array = np.array(boxes)
                    scores_array = np.array(scores)
                    
                    # Convert to format needed for NMS: [x, y, w, h]
                    boxes_xywh = []
                    for box in boxes_array:
                        x1, y1, x2, y2 = box
                        x = x1 * vid_width
                        y = y1 * vid_height
                        w = (x2 - x1) * vid_width
                        h = (y2 - y1) * vid_height
                        boxes_xywh.append([x, y, w, h])
                    
                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores_array.tolist(), confidence_threshold, 0.45)
                    
                    if len(indices) > 0:
                        indices = indices.flatten()
                        boxes = [boxes[i] for i in indices]
                        classes = [classes[i] for i in indices]
                        scores = [scores[i] for i in indices]
                    
                    if frame_count == 0:
                        print(f"DEBUG - After NMS: {len(boxes)} detections")
                
                boxes = np.array(boxes) if len(boxes) > 0 else np.array([])
                classes = np.array(classes) if len(classes) > 0 else np.array([])
                scores = np.array(scores) if len(scores) > 0 else np.array([])
                num_detections = len(boxes)
            else:
                print(f"Warning: Unexpected output format with {len(output_details)} outputs")
                boxes = np.array([])
                classes = np.array([])
                scores = np.array([])
                num_detections = 0
            
            # Draw detections on frame
            annotated_frame = frame.copy()
            detections_in_frame = 0
            
            for i in range(num_detections):
                if scores[i] >= confidence_threshold:
                    # Convert normalized coordinates to pixel coordinates
                    if len(output_details) == 4:
                        # Standard format: [ymin, xmin, ymax, xmax]
                        ymin, xmin, ymax, xmax = boxes[i]
                        left = int(xmin * vid_width)
                        top = int(ymin * vid_height)
                        right = int(xmax * vid_width)
                        bottom = int(ymax * vid_height)
                    else:
                        # YOLO format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = boxes[i]
                        left = int(x1 * vid_width)
                        top = int(y1 * vid_height)
                        right = int(x2 * vid_width)
                        bottom = int(y2 * vid_height)
                    
                    # Debug: print box coordinates on first frame
                    if frame_count == 0:
                        print(f"DEBUG - Drawing box {i}: ({left}, {top}) to ({right}, {bottom}), conf={scores[i]:.2f}")
                    
                    # Draw bounding box with thicker line and different color for visibility
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 3)
                    
                    # Draw label
                    label = f"Class {int(classes[i])}: {scores[i]:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    # Background for text
                    cv2.rectangle(annotated_frame, 
                                (left, top - label_size[1] - 10),
                                (left + label_size[0], top),
                                (0, 255, 0), -1)
                    
                    # Text
                    cv2.putText(annotated_frame, label, (left, top - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    detections_in_frame += 1
            
            if frame_count == 0:
                print(f"DEBUG - Drew {detections_in_frame} boxes on frame")
                print(f"DEBUG - Frame shape: {annotated_frame.shape}, Video size: {vid_width}x{vid_height}")
            
            total_detections += detections_in_frame
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Display the frame
            cv2.imshow('Object Detection - Press Q to quit', annotated_frame)
            
            # Print progress
            if frame_count % 30 == 0 or detections_in_frame > 0:
                progress = (frame_count / total_frames) * 100
                print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - Detections: {detections_in_frame}", end='\r')
            
            frame_count += 1
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nStopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {total_detections/max(frame_count, 1):.2f}")
        print(f"Output saved to: {output_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/tflites/best_algeas.tflite"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Run detection with lowered threshold (your model's max confidence was 0.188)
    run_detection_on_video(
        model_path=model_path,
        video_path=video_path,
        confidence_threshold=0.1,  # Lowered from 0.25 to catch detections
        input_size=640
    )