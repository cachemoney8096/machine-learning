import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


def letterbox_image(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to fit new_shape while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_shape[0]/h, new_shape[1]/w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to filter overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def xywh2xyxy(boxes):
    """Convert [x_center, y_center, w, h] -> [x1, y1, x2, y2]"""
    x_c, y_c, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1 = x_c - w/2
    y1 = y_c - h/2
    x2 = x_c + w/2
    y2 = y_c + h/2
    return np.stack([x1, y1, x2, y2], axis=1)


def get_color(class_id):
    """Generate a consistent color for each class."""
    np.random.seed(class_id)
    return tuple(map(int, np.random.randint(0, 255, 3)))


def draw_detections(frame, detections, class_names=None):
    """Draw bounding boxes and labels on the frame."""
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det

        # Convert to native Python int
        x1 = int(round(float(x1)))
        y1 = int(round(float(y1)))
        x2 = int(round(float(x2)))
        y2 = int(round(float(y2)))
        class_id = int(class_id)
        conf = float(conf)

        # Clip coordinates to frame boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = get_color(class_id)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Create label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]} {conf:.2f}"
        else:
            label = f"Class {class_id} {conf:.2f}"

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_label = max(label_h + 5, y1 - 5)
        cv2.rectangle(annotated_frame, (x1, y_label - label_h - 2), 
                     (x1 + label_w + 4, y_label + 2), color, -1)

        # Draw label text
        cv2.putText(annotated_frame, label, (x1 + 2, y_label), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_frame


def run_detection_on_video(model_path, video_path, output_path=None, 
                          conf_thresh=0.25, iou_thresh=0.45, class_names=None):
    """Run object detection on a video file."""
    
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Get model input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    H, W = input_shape[2] if input_shape[2] else 640, input_shape[3] if input_shape[3] else 640
    output_names = [o.name for o in session.get_outputs()]
    
    print(f"Model input shape: {input_shape}")
    print(f"Model expects: {W}x{H}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output video
    if output_path is None:
        output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_detected.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            img, scale, pad_x, pad_y = letterbox_image(frame, (H, W))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_input = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
            img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

            # Inference
            outputs = session.run(output_names, {input_name: img_input})
            
            # Debug: Print output info on first frame
            if frame_idx == 0:
                print(f"\n--- Model Output Debug Info ---")
                print(f"Number of outputs: {len(outputs)}")
                for i, output in enumerate(outputs):
                    print(f"Output {i} shape: {output.shape}")
                    print(f"Output {i} min/max: {output.min():.4f} / {output.max():.4f}")
                print(f"-------------------------------\n")
            
            preds = outputs[0]
            
            # Handle different output shapes
            if len(preds.shape) == 3:
                preds = preds[0]  # Remove batch dimension if present
            
            # Transpose if needed (some models output [num_classes, num_detections])
            if preds.shape[0] < preds.shape[1] and preds.shape[0] < 100:
                preds = preds.T
            
            # Debug: Print prediction info on first frame
            if frame_idx == 0:
                print(f"Predictions shape after processing: {preds.shape}")
                if len(preds) > 0:
                    print(f"First prediction sample: {preds[0]}")
                    print(f"Predictions min/max: {preds.min():.4f} / {preds.max():.4f}")
            
            if len(preds) == 0:
                detections = []
            else:
                # Extract components
                boxes_xywh = preds[:, :4]
                objectness = preds[:, 4]
                
                # Handle class scores
                if preds.shape[1] > 5:
                    class_scores = preds[:, 5:]
                    class_id = np.argmax(class_scores, axis=1)
                    class_conf = np.max(class_scores, axis=1)
                    conf = objectness * class_conf
                else:
                    conf = objectness
                    class_id = np.zeros(len(preds), dtype=int)
                
                # Filter by confidence
                mask = conf > conf_thresh
                if not np.any(mask):
                    detections = []
                    if frame_idx == 0:
                        print(f"No detections above threshold {conf_thresh}")
                        print(f"Max confidence in frame: {conf.max():.4f}")
                else:
                    boxes_xywh = boxes_xywh[mask]
                    conf = conf[mask]
                    class_id = class_id[mask]
                    
                    # Convert to xyxy format
                    boxes = xywh2xyxy(boxes_xywh)
                    
                    # Scale boxes back to original frame coordinates
                    # First undo padding
                    boxes[:, [0, 2]] -= pad_x
                    boxes[:, [1, 3]] -= pad_y
                    
                    # Then undo scaling
                    boxes /= scale
                    
                    # Clip to frame boundaries
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
                    
                    # Apply NMS
                    keep = non_max_suppression(boxes, conf, iou_thresh)
                    
                    # Format detections
                    detections = [[boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3], 
                                 conf[i], class_id[i]] for i in keep]
                    
                    if frame_idx == 0 and len(detections) > 0:
                        print(f"Found {len(detections)} detections after NMS")
                        print(f"First detection: {detections[0]}")

            # Draw detections and save frame
            out_frame = draw_detections(frame, detections, class_names)
            out.write(out_frame)
            
            # Display (optional - comment out for faster processing)
            cv2.imshow("Detection", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
            
            frame_idx += 1
            if frame_idx % 30 == 0:  # Update every 30 frames
                print(f"\rProcessing: {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)", end='')
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"\n\nDone! Output saved to {output_path}")
    print(f"Processed {frame_idx} frames")


if __name__ == "__main__":
    # Define your class names here (optional)
    class_names = ["coral"]  # Update with your actual class names
    
    # Update these paths to your files
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v16/weights/best.onnx"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Run detection
    run_detection_on_video(
        model_path=model_path,
        video_path=video_path,
        conf_thresh=0.25,  # Adjust confidence threshold as needed
        iou_thresh=0.45,
        class_names=class_names
    )