import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def run_detection_on_video(model_path, video_path, output_path=None, confidence_threshold=0.25):
    """
    Run object detection on every frame of a video using an ONNX model file.
    
    Args:
        model_path: Path to .onnx model file
        video_path: Path to input video file
        output_path: Path for output video (optional)
        confidence_threshold: Minimum confidence for detections (0-1)
    """
    # Load the ONNX model
    print(f"Loading ONNX model from {model_path}...")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get model input details
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        input_shape = model_inputs[0].shape
        
        # Get expected input size (typically [batch, channels, height, width])
        if len(input_shape) == 4:
            input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
            input_width = input_shape[3] if isinstance(input_shape[3], int) else 640
        else:
            input_height = input_width = 640  # Default size
        
        print(f"Model loaded successfully! Input size: {input_width}x{input_height}")
        
        # Get output details
        output_names = [output.name for output in session.get_outputs()]
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Set output path if not provided
    if output_path is None:
        input_file = Path(video_path)
        output_path = input_file.parent / f"{input_file.stem}_detected{input_file.suffix}"
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("\nProcessing video frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess frame for ONNX model
            original_frame = frame.copy()
            
            # Resize and normalize
            input_frame = cv2.resize(frame, (input_width, input_height))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = input_frame.astype(np.float32) / 255.0
            
            # Change from HWC to CHW format
            input_frame = np.transpose(input_frame, (2, 0, 1))
            
            # Add batch dimension
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Run inference
            outputs = session.run(output_names, {input_name: input_frame})
            
            # Post-process detections
            detections = post_process_detections(
                outputs, 
                original_frame.shape, 
                (input_width, input_height),
                confidence_threshold
            )
            
            # Draw bounding boxes
            annotated_frame = draw_detections(original_frame, detections)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Display the frame
            cv2.imshow('Object Detection', annotated_frame)
            
            # Print detection info
            if len(detections) > 0:
                print(f"\rFrame {frame_count}/{total_frames}: {len(detections)} objects detected", end='')
            else:
                print(f"\rFrame {frame_count}/{total_frames}: 0 objects detected", end='')
            
            frame_count += 1
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Processed {frame_count} frames")


def post_process_detections(outputs, original_shape, input_shape, conf_threshold):
    """
    Post-process ONNX model outputs to extract bounding boxes.
    Supports YOLOv5/YOLOv8 ONNX format.
    
    Args:
        outputs: Raw model outputs
        original_shape: Original frame shape (height, width, channels)
        input_shape: Model input shape (width, height)
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections [x1, y1, x2, y2, confidence, class_id]
    """
    predictions = outputs[0]
    
    # Handle different output formats
    if len(predictions.shape) == 3:
        predictions = predictions[0]
    
    # Filter by confidence
    if predictions.shape[1] > 5:
        # Format: [x, y, w, h, conf, class_scores...]
        confidences = predictions[:, 4]
        mask = confidences > conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        # Get class with highest score
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # Combine objectness and class confidence
        confidences = predictions[:, 4] * class_confidences
    else:
        # Format: [x, y, w, h, conf] (single class or already processed)
        confidences = predictions[:, 4]
        mask = confidences > conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        class_ids = np.zeros(len(predictions), dtype=int)
    
    # Convert from xywh to xyxy and scale to original image size
    boxes = predictions[:, :4]
    
    # Scale boxes to original image size
    orig_h, orig_w = original_shape[:2]
    input_w, input_h = input_shape
    
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    
    # Convert center format to corner format if needed
    if boxes.max() <= 1.0:
        # Normalized coordinates
        boxes[:, [0, 2]] *= input_w
        boxes[:, [1, 3]] *= input_h
    
    # xywh to xyxy
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_center - w / 2) * scale_x
    y1 = (y_center - h / 2) * scale_y
    x2 = (x_center + w / 2) * scale_x
    y2 = (y_center + h / 2) * scale_y
    
    # Apply NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = non_max_suppression(boxes_xyxy, confidences, iou_threshold=0.45)
    
    # Format detections
    detections = []
    for idx in indices:
        detections.append([
            int(boxes_xyxy[idx, 0]),
            int(boxes_xyxy[idx, 1]),
            int(boxes_xyxy[idx, 2]),
            int(boxes_xyxy[idx, 3]),
            float(confidences[idx]),
            int(class_ids[idx])
        ])
    
    return detections


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.
    """
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def draw_detections(frame, detections, class_names=None):
    """
    Draw bounding boxes and labels on the frame.
    """
    annotated_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        
        # Generate color based on class_id
        color = get_color(class_id)
        
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]} {conf:.2f}"
        else:
            label = f"Class {class_id} {conf:.2f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_frame


def get_color(class_id):
    """
    Generate a consistent color for each class.
    """
    np.random.seed(class_id)
    return tuple(map(int, np.random.randint(0, 255, 3)))


if __name__ == "__main__":
    # Example usage
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best.onnx"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Run detection with default settings
    run_detection_on_video(
        model_path=model_path,
        video_path=video_path,
        confidence_threshold=0.25  # Adjust confidence threshold as needed
    )
    
    # Or specify output path
    # run_detection_on_video(model_path, video_path, output_path="output.mp4")