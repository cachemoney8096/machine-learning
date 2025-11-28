import cv2
import numpy as np
from pathlib import Path

# Import for PT model
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available for .pt model")
    YOLO_AVAILABLE = False

# Import for TFLite model
try:
    import tensorflow as tf
    if hasattr(tf, 'lite'):
        Interpreter = tf.lite.Interpreter
    else:
        raise ImportError
    TFLITE_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        TFLITE_AVAILABLE = True
    except ImportError:
        print("Warning: TFLite not available")
        TFLITE_AVAILABLE = False


def compare_models_on_video(pt_path, tflite_path, video_path, output_path=None, 
                            confidence_threshold=0.1, num_frames=None):
    """
    Compare .pt and .tflite models side-by-side on video frames.
    
    Args:
        pt_path: Path to .pt model file
        tflite_path: Path to .tflite model file
        video_path: Path to input video file
        output_path: Path for comparison video (optional)
        confidence_threshold: Minimum confidence for detections
        num_frames: Number of frames to process (None = all frames)
    """
    
    # Load PT model
    print("Loading PyTorch model...")
    if not YOLO_AVAILABLE:
        print("ERROR: Cannot load .pt model - ultralytics not installed")
        return
    
    pt_model = YOLO(pt_path)
    pt_model.conf = confidence_threshold
    print("✓ PT model loaded")
    
    # Load TFLite model
    print("\nLoading TFLite model...")
    if not TFLITE_AVAILABLE:
        print("ERROR: Cannot load .tflite model - TFLite not installed")
        return
    
    tflite_interpreter = Interpreter(model_path=tflite_path)
    tflite_interpreter.allocate_tensors()
    
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    model_height = input_shape[1]
    model_width = input_shape[2]
    
    print(f"✓ TFLite model loaded")
    print(f"  Input shape: {input_shape}")
    print(f"  Outputs: {len(output_details)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames:
        total_frames = min(num_frames, total_frames)
    
    print(f"\nVideo: {width}x{height} @ {fps}fps, processing {total_frames} frames")
    
    # Setup output video (side-by-side comparison)
    if output_path is None:
        output_path = Path(video_path).parent / f"{Path(video_path).stem}_comparison.mp4"
    
    # Create side-by-side output (double width)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))
    
    # Statistics tracking
    pt_total_detections = 0
    tflite_total_detections = 0
    matching_detections = 0
    frame_count = 0
    
    print("\nProcessing frames...")
    print("=" * 80)
    
    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ===== PyTorch Model Inference =====
            pt_results = pt_model(frame, verbose=False)
            pt_frame = pt_results[0].plot()
            pt_detections = len(pt_results[0].boxes)
            pt_total_detections += pt_detections
            
            # Get PT detection details
            pt_boxes = []
            for box in pt_results[0].boxes:
                pt_boxes.append({
                    'class': int(box.cls),
                    'conf': float(box.conf),
                    'bbox': box.xyxy[0].cpu().numpy()
                })
            
            # ===== TFLite Model Inference =====
            # Preprocess
            input_data = cv2.resize(frame, (model_width, model_height))
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_data, axis=0)
            
            if input_details[0]['dtype'] == np.float32:
                input_data = input_data.astype(np.float32) / 255.0
            else:
                input_data = input_data.astype(np.uint8)
            
            # Run inference
            tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            
            # Parse TFLite output
            tflite_frame = frame.copy()
            tflite_boxes = []
            
            if len(output_details) == 4:
                # Standard detection format
                boxes = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
                classes = tflite_interpreter.get_tensor(output_details[1]['index'])[0]
                scores = tflite_interpreter.get_tensor(output_details[2]['index'])[0]
                num_det = int(tflite_interpreter.get_tensor(output_details[3]['index'])[0])
                
                for i in range(num_det):
                    if scores[i] >= confidence_threshold:
                        ymin, xmin, ymax, xmax = boxes[i]
                        x1 = int(xmin * width)
                        y1 = int(ymin * height)
                        x2 = int(xmax * width)
                        y2 = int(ymax * height)
                        
                        tflite_boxes.append({
                            'class': int(classes[i]),
                            'conf': float(scores[i]),
                            'bbox': np.array([x1, y1, x2, y2])
                        })
                        
                        # Draw on frame
                        cv2.rectangle(tflite_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Class {int(classes[i])}: {scores[i]:.2f}"
                        cv2.putText(tflite_frame, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif len(output_details) == 1:
                # YOLO format - CORRECTED VERSION
                output = tflite_interpreter.get_tensor(output_details[0]['index'])
                
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
                
                # Parse detections
                for detection in output:
                    if len(detection) >= 5:
                        x_center, y_center, w, h, confidence = detection[0:5]
                        
                        if confidence >= confidence_threshold:
                            # For single-class models, class_id is 0
                            if len(detection) == 5:
                                class_id = 0
                                class_conf = 1.0
                            else:
                                # Multi-class: get class with highest score
                                class_scores = detection[5:]
                                class_id = np.argmax(class_scores)
                                class_conf = class_scores[class_id]
                            
                            # YOLO outputs coordinates relative to input size
                            # Convert to pixel coordinates in original frame
                            x1 = int((x_center - w/2) * width / model_width)
                            y1 = int((y_center - h/2) * height / model_height)
                            x2 = int((x_center + w/2) * width / model_width)
                            y2 = int((y_center + h/2) * height / model_height)
                            
                            # Clamp to frame boundaries
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(width, x2), min(height, y2)
                            
                            tflite_boxes.append({
                                'class': int(class_id),
                                'conf': float(confidence),
                                'bbox': np.array([x1, y1, x2, y2])
                            })
                            
                            # Draw on frame
                            cv2.rectangle(tflite_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"Class {class_id}: {confidence:.2f}"
                            cv2.putText(tflite_frame, label, (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Apply NMS to remove duplicates
                if len(tflite_boxes) > 0:
                    tflite_boxes = apply_nms(tflite_boxes, confidence_threshold, 0.45)
                    
                    # Redraw after NMS
                    tflite_frame = frame.copy()
                    for box_data in tflite_boxes:
                        x1, y1, x2, y2 = box_data['bbox'].astype(int)
                        cv2.rectangle(tflite_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Class {box_data['class']}: {box_data['conf']:.2f}"
                        cv2.putText(tflite_frame, label, (x1, y1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            tflite_detections = len(tflite_boxes)
            tflite_total_detections += tflite_detections
            
            # Count matching detections (IoU > 0.5)
            matches = count_matching_boxes(pt_boxes, tflite_boxes)
            matching_detections += matches
            
            # Add labels to frames
            pt_label = f"PyTorch: {pt_detections} detections"
            tflite_label = f"TFLite: {tflite_detections} detections"
            match_label = f"Matches: {matches}"
            
            cv2.putText(pt_frame, pt_label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(tflite_frame, tflite_label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Create side-by-side comparison
            comparison = np.hstack((pt_frame, tflite_frame))
            
            # Add comparison stats at the top
            cv2.putText(comparison, match_label, (width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write to output
            out.write(comparison)
            
            # Display
            cv2.imshow('Comparison: PT (left) vs TFLite (right) - Press Q to quit', comparison)
            
            # Print frame stats
            if frame_count % 30 == 0 or pt_detections != tflite_detections:
                diff = abs(pt_detections - tflite_detections)
                status = "✓" if diff == 0 else "⚠"
                print(f"{status} Frame {frame_count}/{total_frames}: PT={pt_detections}, TFLite={tflite_detections}, Diff={diff}")
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Frames processed: {frame_count}")
        print(f"\nTotal detections:")
        print(f"  PyTorch:  {pt_total_detections}")
        print(f"  TFLite:   {tflite_total_detections}")
        print(f"  Difference: {abs(pt_total_detections - tflite_total_detections)}")
        print(f"\nAverage detections per frame:")
        print(f"  PyTorch:  {pt_total_detections/max(frame_count, 1):.2f}")
        print(f"  TFLite:   {tflite_total_detections/max(frame_count, 1):.2f}")
        print(f"\nMatching detections: {matching_detections}/{max(pt_total_detections, tflite_total_detections)} "
              f"({100*matching_detections/max(pt_total_detections, tflite_total_detections, 1):.1f}%)")
        print(f"\nOutput saved to: {output_path}")
        print("=" * 80)


def apply_nms(boxes, conf_threshold, iou_threshold):
    """Apply Non-Maximum Suppression to remove duplicate detections."""
    if len(boxes) == 0:
        return []
    
    # Extract data for NMS
    boxes_xywh = []
    scores = []
    for box_data in boxes:
        x1, y1, x2, y2 = box_data['bbox']
        boxes_xywh.append([x1, y1, x2-x1, y2-y1])
        scores.append(box_data['conf'])
    
    # Apply OpenCV NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices]
    
    return []


def count_matching_boxes(boxes1, boxes2, iou_threshold=0.5):
    """Count boxes that match between two sets based on IoU and class."""
    matches = 0
    
    for box1 in boxes1:
        for box2 in boxes2:
            # Check if same class
            if box1['class'] != box2['class']:
                continue
            
            # Calculate IoU
            iou = calculate_iou(box1['bbox'], box2['bbox'])
            
            if iou >= iou_threshold:
                matches += 1
                break  # Found a match for box1
    
    return matches


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    intersection = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


if __name__ == "__main__":
    # Your paths
    pt_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best.pt"
    tflite_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best_float32.tflite"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Run comparison with lowered threshold (max confidence was 0.188)
    compare_models_on_video(
        pt_path=pt_path,
        tflite_path=tflite_path,
        video_path=video_path,
        confidence_threshold=0.1,  # Lowered from 0.25
        num_frames=None  # Set to a number like 100 to test on first 100 frames only
    )