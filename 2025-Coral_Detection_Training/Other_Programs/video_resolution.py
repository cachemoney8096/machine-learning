import cv2
import os
from pathlib import Path

def resize_video_to_640x640(input_path, output_path=None, maintain_aspect=True):
    """
    Convert all frames of an MP4 video to 640x640 resolution.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path for output MP4 file (optional)
        maintain_aspect: If True, maintains aspect ratio with padding. 
                        If False, stretches to 640x640
    """
    # Set output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_640x640{input_file.suffix}"
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 640))
    
    print(f"Processing {total_frames} frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if maintain_aspect:
            # Resize maintaining aspect ratio with padding
            h, w = frame.shape[:2]
            
            # Calculate scaling factor
            scale = min(640 / w, 640 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create black canvas and center the resized frame
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            y_offset = (640 - new_h) // 2
            x_offset = (640 - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            out.write(canvas)
        else:
            # Stretch to 640x640
            resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LANCZOS4)
            out.write(resized)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\nConversion complete!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    import numpy as np
    
    # Example usage
    input_video = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/people_walking.mp4"
    
    # Option 1: Maintain aspect ratio (adds black bars if needed)
    resize_video_to_640x640(input_video, maintain_aspect=True)
    
    # Option 2: Stretch to 640x640 (may distort video)
    # resize_video_to_640x640(input_video, maintain_aspect=False)