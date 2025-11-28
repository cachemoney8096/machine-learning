import cv2
import torch
from pathlib import Path

def run_detection_on_video(model_path, video_path, output_path=None, confidence_threshold=0.25):
    """
    Run object detection on every frame of a video using a .pt model file.
    
    Args:
        model_path: Path to .pt model file
        video_path: Path to input video file
        output_path: Path for output video (optional)
        confidence_threshold: Minimum confidence for detections (0-1)
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        # For YOLOv5/YOLOv8 models
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.conf = confidence_threshold
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print("Model loaded successfully with ultralytics!")
        except Exception as e2:
            print(f"Error: {e2}")
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
            
            # Run detection
            results = model(frame)
            
            # Draw results on frame
            if hasattr(results, 'render'):
                # YOLOv5 style
                annotated_frame = results.render()[0]
            elif hasattr(results[0], 'plot'):
                # YOLOv8 style
                annotated_frame = results[0].plot()
            else:
                # Fallback: draw manually
                annotated_frame = frame.copy()
                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Display the frame
            cv2.imshow('Object Detection', annotated_frame)
            
            # Print detections info
            if hasattr(results, 'pandas'):
                detections = results.pandas().xyxy[0]
                if len(detections) > 0:
                    print(f"\rFrame {frame_count}/{total_frames}: {len(detections)} objects detected", end='')
            
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


if __name__ == "__main__":
    # Example usage
    model_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral-detection-model-v113/weights/best.pt"
    video_path = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/frc_640x640.mp4"
    
    # Run detection with default settings
    run_detection_on_video(
        model_path=model_path,
        video_path=video_path,
        confidence_threshold=0.25  # Adjust confidence threshold as needed
    )
    
    # Or specify output path
    # run_detection_on_video(model_path, video_path, output_path="output.mp4")