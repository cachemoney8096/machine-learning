import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf


def run_detection_on_video(model_path, video_path, output_path=None, 
                          input_size=640, class_names=None, confidence_threshold=0.25):
    """
    Run object detection on video using TFLite model.
    Automatically detects output format (YOLO or TFOD style).
    """
    # Load the TFLite model
    print(f"Loading TFLite model from {model_path}...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("✓ Model loaded successfully!")
        print(f"Input shape: {input_details[0]['shape']}")
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"\nNumber of outputs: {len(output_details)}")
        print("Output details:")
        for i, od in enumerate(output_details):
            print(f"  Output {i}: shape={od['shape']}, dtype={od['dtype']}, name={od.get('name', 'N/A')}")
        
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {vid_width}x{vid_height} @ {fps}fps, {total_frames} frames")
    
    # Setup output
    if output_path is None:
        input_file = Path(video_path)
        output_path = input_file.parent / f"{input_file.stem}_detected{input_file.suffix}"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_width, vid_height))
    
    frame_count = 0
    total_detections = 0
    first_frame_debug = True
    
    print(f"\nProcessing video (confidence threshold: {confidence_threshold})")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            img_resized = cv2.resize(frame, (input_width, input_height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            if input_details[0]['dtype'] == np.float32:
                img_input = img_rgb.astype(np.float32) / 255.0
            else:
                img_input = img_rgb.astype(np.uint8)
            
            img_input = np.expand_dims(img_input, axis=0)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()
            
            # Get all outputs
            outputs = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            
            # Debug first frame
            if first_frame_debug:
                print("\n=== FIRST FRAME DEBUG ===")
                for i, output in enumerate(outputs):
                    print(f"Output {i} shape: {output.shape}")
                    print(f"Output {i} dtype: {output.dtype}")
                    print(f"Output {i} min/max: {output.min():.4f} / {output.max():.4f}")
                    if len(output.shape) <= 3:
                        print(f"Output {i} sample values: {output.flatten()[:10]}")
                    print()
                first_frame_debug = False
                print("======================\n")
            
            # Debug detections per class
            detections_by_class = {}
            
            # Parse outputs based on structure
            final_boxes = []
            final_scores = []
            final_class_ids = []
            
            # Detect output format
            if len(outputs) == 1:
                # YOLO-style: single output
                output = outputs[0]
                
                # Remove batch dimension
                if len(output.shape) == 3:
                    output = output[0]
                
                print(f"Single output shape after squeeze: {output.shape}")
                
                # Common YOLO formats:
                # (8400, 6) or (N, 6) = [x, y, w, h, conf, class]
                # (8400, 85) or (N, 85) = [x, y, w, h, conf, 80 classes]
                # (6, 8400) or (85, 8400) = transposed version
                
                # Check if transposed
                if output.shape[0] < output.shape[1] and output.shape[0] <= 85:
                    output = output.T
                    print(f"Transposed to: {output.shape}")
                
                num_predictions = output.shape[0]
                num_values = output.shape[1]
                
                print(f"Processing {num_predictions} predictions with {num_values} values each")
                
                # Parse based on number of values
                if num_values == 4:
                    # Just boxes [x, y, w, h] - no confidence
                    boxes_xywh = output
                    confidences = np.ones(num_predictions)
                    class_ids = np.zeros(num_predictions, dtype=int)
                elif num_values == 5:
                    # [x, y, w, h, class_id]
                    boxes_xywh = output[:, :4]
                    confidences = np.ones(num_predictions)
                    class_ids = output[:, 4].astype(int)
                elif num_values == 6:
                    # [x, y, w, h, conf, class_id]
                    boxes_xywh = output[:, :4]
                    confidences = output[:, 4]
                    class_ids = output[:, 5].astype(int)
                elif num_values >= 85:
                    # [x, y, w, h, obj_conf, class1, class2, ...]
                    boxes_xywh = output[:, :4]
                    obj_conf = output[:, 4]
                    class_scores = output[:, 5:]
                    class_ids = np.argmax(class_scores, axis=1)
                    class_confidences = np.max(class_scores, axis=1)
                    confidences = obj_conf * class_confidences
                else:
                    # Assume [x, y, w, h, class_scores...]
                    boxes_xywh = output[:, :4]
                    class_scores = output[:, 4:]
                    class_ids = np.argmax(class_scores, axis=1)
                    confidences = np.max(class_scores, axis=1)
                
                # Filter by confidence and process boxes
                for i in range(num_predictions):
                    conf = float(confidences[i])
                    class_id = int(class_ids[i])
                    
                    # Debug: count all detections before filtering
                    if class_id not in detections_by_class:
                        detections_by_class[class_id] = {'total': 0, 'filtered': 0}
                    detections_by_class[class_id]['total'] += 1
                    
                    if conf < confidence_threshold:
                        continue
                    
                    detections_by_class[class_id]['filtered'] += 1
                    
                    # Get box coordinates
                    x_center, y_center, w, h = boxes_xywh[i]
                    
                    # Check if coordinates are normalized (0-1) or pixel values
                    if x_center <= 1.0 and y_center <= 1.0 and w <= 1.0 and h <= 1.0:
                        # Normalized coordinates
                        x1 = (x_center - w/2) * vid_width
                        y1 = (y_center - h/2) * vid_height
                        x2 = (x_center + w/2) * vid_width
                        y2 = (y_center + h/2) * vid_height
                    else:
                        # Pixel coordinates relative to input size
                        scale_x = vid_width / input_width
                        scale_y = vid_height / input_height
                        x1 = (x_center - w/2) * scale_x
                        y1 = (y_center - h/2) * scale_y
                        x2 = (x_center + w/2) * scale_x
                        y2 = (y_center + h/2) * scale_y
                    
                    x1 = int(max(0, min(x1, vid_width - 1)))
                    y1 = int(max(0, min(y1, vid_height - 1)))
                    x2 = int(max(0, min(x2, vid_width - 1)))
                    y2 = int(max(0, min(y2, vid_height - 1)))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    final_boxes.append([x1, y1, x2, y2])
                    final_scores.append(conf)
                    final_class_ids.append(int(class_ids[i]))
                
                # Print class distribution every 30 frames
                if frame_count % 30 == 0 and detections_by_class:
                    print(f"\n--- Frame {frame_count} Class Distribution ---")
                    for cls_id, counts in sorted(detections_by_class.items()):
                        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
                        print(f"{cls_name}: {counts['filtered']}/{counts['total']} passed threshold (conf > {confidence_threshold})")
                    print()
            
            elif len(outputs) == 4:
                # TFOD-style: [num_detections, boxes, scores, classes]
                num_det_tensor = outputs[0]
                boxes = outputs[1]
                scores = outputs[2]
                classes = outputs[3]
                
                # Handle num_detections
                if num_det_tensor.size == 1:
                    num_detections = int(num_det_tensor)
                else:
                    num_detections = boxes.shape[1] if len(boxes.shape) > 1 else 10
                
                # Remove batch dimension
                if len(boxes.shape) == 3:
                    boxes = boxes[0]
                if len(scores.shape) == 2:
                    scores = scores[0]
                if len(classes.shape) == 2:
                    classes = classes[0]
                
                for i in range(min(num_detections, len(scores))):
                    score = float(scores[i])
                    class_id = int(classes[i])
                    
                    # Debug: count all detections before filtering
                    if class_id not in detections_by_class:
                        detections_by_class[class_id] = {'total': 0, 'filtered': 0}
                    detections_by_class[class_id]['total'] += 1
                    
                    if score < confidence_threshold:
                        continue
                    
                    detections_by_class[class_id]['filtered'] += 1
                    
                    # TFOD format: [ymin, xmin, ymax, xmax] normalized
                    ymin, xmin, ymax, xmax = boxes[i]
                    
                    x1 = int(xmin * vid_width)
                    y1 = int(ymin * vid_height)
                    x2 = int(xmax * vid_width)
                    y2 = int(ymax * vid_height)
                    
                    x1 = max(0, min(x1, vid_width - 1))
                    y1 = max(0, min(y1, vid_height - 1))
                    x2 = max(0, min(x2, vid_width - 1))
                    y2 = max(0, min(y2, vid_height - 1))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    final_boxes.append([x1, y1, x2, y2])
                    final_scores.append(score)
                    final_class_ids.append(int(classes[i]))
                
                # Print class distribution every 30 frames
                if frame_count % 30 == 0 and detections_by_class:
                    print(f"\n--- Frame {frame_count} Class Distribution ---")
                    for cls_id, counts in sorted(detections_by_class.items()):
                        cls_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
                        print(f"{cls_name}: {counts['filtered']}/{counts['total']} passed threshold (conf > {confidence_threshold})")
                    print()
            
            total_detections += len(final_boxes)
            
            # Draw detections
            annotated_frame = frame.copy()
            
            for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                if class_id == 0:
                    color = (0, 255, 0)
                    class_name = class_names[0] if class_names else "Class 0"
                else:
                    color = (255, 0, 0)
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                
                thickness = 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"{class_name} {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 5, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness
                )
            
            out.write(annotated_frame)
            cv2.imshow('Detection - Press Q to quit', annotated_frame)
            
            if frame_count % 30 == 0 or len(final_boxes) > 0:
                progress = (frame_count + 1) / total_frames * 100
                avg_det = total_detections / (frame_count + 1)
                print(
                    f"Frame {frame_count + 1}/{total_frames} ({progress:.1f}%) | "
                    f"Detections: {len(final_boxes)} | Avg: {avg_det:.2f}",
                    end='\r'
                )
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nStopped by user")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average per frame: {total_detections/max(frame_count, 1):.2f}")
        print(f"Output saved: {output_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v112/weights/best_saved_model/best_float32.tflite"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    class_names = ["Algae", "Coral"]
    
    run_detection_on_video(
        model_path=model_path,
        video_path=video_path,
        input_size=640,
        class_names=class_names,
        confidence_threshold=0.1
    )