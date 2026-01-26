import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd

# Use ultralytics utilities if available
try:
    from ultralytics import YOLO
    from ultralytics.utils import ops
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("⚠️  Ultralytics not available - some optimizations disabled")

# Use torchvision for NMS if available
try:
    from torchvision.ops import nms, box_iou
    import torch
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ============================================================================
# CONFIGURATION
# ============================================================================
CURRENT_TFLITE_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v112/weights/best_saved_model/best_float32.tflite"
REFERENCE_TFLITE_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v112/weights/detect_quant.tflite"
ORIGINAL_PT_MODEL = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best.pt"
TEST_IMAGES_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/calib_images/*"
OUTPUT_FOLDER = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/comparison_results"

CONFIDENCE_THRESHOLD = 0.3
CLASS_NAMES = ["Algae", "Coral"]

# ============================================================================
# DATA STRUCTURES
# ============================================================================
@dataclass
class Detection:
    """Standard detection format"""
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    
    @property
    def xyxy(self) -> np.ndarray:
        """Return box in xyxy format as numpy array"""
        return np.array(self.box)
    
    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id] if self.class_id < len(CLASS_NAMES) else f"Class {self.class_id}"


# ============================================================================
# UTILITY FUNCTIONS USING PACKAGES
# ============================================================================
def apply_nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Apply Non-Maximum Suppression using available packages"""
    if not detections:
        return detections
    
    if HAS_TORCHVISION:
        # Use torchvision NMS (fastest)
        boxes = torch.tensor([d.xyxy for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.confidence for d in detections], dtype=torch.float32)
        keep_indices = nms(boxes, scores, iou_threshold)
        return [detections[i] for i in keep_indices.numpy()]
    else:
        # Fallback to cv2.dnn.NMSBoxes
        boxes = [d.box for d in detections]
        scores = [d.confidence for d in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []


def letterbox_image(img: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox image for model input (maintains aspect ratio with padding)
    Uses cv2 for resizing
    """
    h, w = img.shape[:2]
    new_h, new_w = new_shape
    
    # Calculate scaling factor
    scale = min(new_h / h, new_w / w)
    scaled_h, scaled_w = int(h * scale), int(w * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    img_padded = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    pad_top = (new_h - scaled_h) // 2
    pad_left = (new_w - scaled_w) // 2
    
    img_padded[pad_top:pad_top + scaled_h, pad_left:pad_left + scaled_w] = img_resized
    
    return img_padded, scale, (pad_left, pad_top)


def preprocess_image(img: np.ndarray, input_shape: Tuple[int, int], 
                     dtype: np.dtype, use_rgb: bool = True) -> np.ndarray:
    """Unified image preprocessing"""
    img_resized = cv2.resize(img, input_shape)
    
    if use_rgb:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    if dtype == np.float32:
        img_input = img_resized.astype(np.float32) / 255.0
    else:
        img_input = img_resized.astype(dtype)
    
    return np.expand_dims(img_input, axis=0)


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================
def detect_yolo_tflite(interpreter, input_details, output_details, 
                       img: np.ndarray, conf_threshold: float) -> List[Detection]:
    """YOLO-style TFLite detection using package utilities"""
    h, w = img.shape[:2]
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    
    # Preprocess
    img_input = preprocess_image(img, (input_width, input_height), 
                                 input_details[0]['dtype'], use_rgb=True)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # DEBUG: Print output shape and sample values
    print(f"\n🔍 DEBUG - Raw output shape: {output.shape}")
    print(f"   Output tensor dtype: {output_details[0]['dtype']}")
    
    # Check if there are multiple output tensors
    if len(output_details) > 1:
        print(f"   ⚠️ Model has {len(output_details)} output tensors!")
        for i in range(len(output_details)):
            extra_output = interpreter.get_tensor(output_details[i]['index'])
            print(f"      Output {i}: shape {extra_output.shape}, dtype {output_details[i]['dtype']}")
    
    if len(output.shape) == 3:
        output = output[0]
    
    # Transpose if needed
    if output.shape[0] < output.shape[1] and output.shape[0] <= 85:
        print(f"   Transposing from {output.shape} to {output.T.shape}")
        output = output.T
    
    print(f"   Final output shape: {output.shape}")
    print(f"   Number of predictions: {output.shape[0]}")
    
    # Show a sample prediction
    if output.shape[0] > 0:
        sample_pred = output[np.argmax(output[:, 4])]  # Highest confidence prediction
        print(f"   Sample prediction (highest conf): {sample_pred}")
        print(f"      -> [x={sample_pred[0]:.3f}, y={sample_pred[1]:.3f}, w={sample_pred[2]:.3f}, h={sample_pred[3]:.3f}, conf={sample_pred[4]:.3f}, val5={sample_pred[5]:.6f}]")
    
    # Parse detections using numpy operations
    detections = []
    num_values = output.shape[1]
    print(f"   Values per prediction: {num_values}")
    
    # Check all output tensors for multi-output models
    if len(output_details) > 1:
        print(f"   ⚠️ Model has {len(output_details)} output tensors!")
        for i, od in enumerate(output_details):
            extra_output = interpreter.get_tensor(od['index'])
            print(f"      Output {i}: shape {extra_output.shape}, dtype {od['dtype']}")
            if i < 3:  # Show sample of first few
                print(f"         Sample: {extra_output.flatten()[:10]}")
    
    if num_values == 6:
        # Could be two formats:
        # Format 1: [x, y, w, h, conf, class_id] - class_id should be 0 or 1
        # Format 2: [x, y, w, h, class0_score, class1_score] - need argmax
        boxes_xywh = output[:, :4]
        
        # Check if last two columns are class scores
        val5 = output[:, 5]
        
        # If values are all between 0-1 and never exactly 0 or 1, likely class scores
        if len(output_details) == 1 and output.shape[1] == 6:
            # Try interpretation: columns 4 and 5 might be class scores for 2 classes
            class0_scores = output[:, 4]
            class1_scores = output[:, 5]
            
            print(f"   Format: Testing if cols 4&5 are class scores")
            print(f"   Column 4 (might be class 0 - Algae): min={class0_scores.min():.6f}, max={class0_scores.max():.6f}")
            print(f"   Column 5 (might be class 1 - Coral): min={class1_scores.min():.6f}, max={class1_scores.max():.6f}")
            
            # Stack class scores and find max
            class_scores = np.stack([class0_scores, class1_scores], axis=1)
            class_ids = np.argmax(class_scores, axis=1)
            confidences = np.max(class_scores, axis=1)
            
            print(f"   Unique class IDs from argmax: {np.unique(class_ids)}")
            print(f"   Max confidence: {confidences.max():.4f}")
            print(f"   Class distribution: Class 0 (Algae)={np.sum(class_ids==0)}, Class 1 (Coral)={np.sum(class_ids==1)}")
        else:
            # Original interpretation
            confidences = output[:, 4]
            class_ids = output[:, 5].astype(int)
            print(f"   Format: 6-value (xywh, conf, class)")
            print(f"   Max confidence: {confidences.max():.4f}")
            print(f"   Classes present: {np.unique(class_ids)}")
    elif num_values >= 85:
        # Format: [x, y, w, h, obj_conf, class1, class2, ...]
        boxes_xywh = output[:, :4]
        obj_conf = output[:, 4]
        class_scores = output[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = obj_conf * np.max(class_scores, axis=1)
        print(f"   Format: 85+ value (xywh, obj_conf, {class_scores.shape[1]} classes)")
        print(f"   Max obj_conf: {obj_conf.max():.4f}")
        print(f"   Max class_score: {class_scores.max():.4f}")
        print(f"   Max final confidence: {confidences.max():.4f}")
        print(f"   Classes present: {np.unique(class_ids)}")
        # Debug: show class scores for highest confidence detection
        max_idx = np.argmax(confidences)
        print(f"   Best detection class scores: {class_scores[max_idx][:10]}")
    else:
        # Format: [x, y, w, h, class1, class2, ...]
        boxes_xywh = output[:, :4]
        class_scores = output[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        print(f"   Format: {num_values}-value (xywh, {class_scores.shape[1]} classes)")
        print(f"   Max confidence: {confidences.max():.4f}")
        print(f"   Classes present: {np.unique(class_ids)}")
        # Debug: show class scores for highest confidence detection
        max_idx = np.argmax(confidences)
        print(f"   Best detection class scores: {class_scores[max_idx]}")
    
    # Filter by confidence
    mask = confidences >= conf_threshold
    print(f"   Detections above {conf_threshold}: {mask.sum()}")
    
    # Early return if no detections pass threshold
    if not mask.any():
        print(f"   ⚠️ No detections above threshold!")
        return []
    
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    print(f"   Filtered classes: {np.unique(class_ids)}")
    
    # Convert xywh to xyxy
    if HAS_ULTRALYTICS:
        # Use ultralytics utility
        boxes_xyxy = ops.xywh2xyxy(boxes_xywh)
    else:
        # Manual conversion
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    
    # Scale to image coordinates (check if we have any boxes first)
    if len(boxes_xyxy) > 0 and boxes_xyxy.max() <= 1.0:
        # Normalized coordinates
        print(f"   Using normalized coordinates")
        boxes_xyxy[:, [0, 2]] *= w
        boxes_xyxy[:, [1, 3]] *= h
    else:
        # Input size coordinates
        print(f"   Using input-size coordinates")
        scale_x, scale_y = w / input_width, h / input_height
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
    
    # Clip to image bounds using numpy
    boxes_xyxy = np.clip(boxes_xyxy, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
    
    # Create Detection objects
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        if x2 > x1 and y2 > y1:
            detections.append(Detection(
                box=[x1, y1, x2, y2],
                confidence=float(confidences[i]),
                class_id=int(class_ids[i])
            ))
    
    print(f"   ✓ Final valid detections: {len(detections)}")
    
    return detections


def detect_tfod_tflite(interpreter, input_details, output_details, 
                       img: np.ndarray, conf_threshold: float) -> List[Detection]:
    """TensorFlow Object Detection API style detection"""
    h, w = img.shape[:2]
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    
    # Preprocess
    img_input = preprocess_image(img, (input_width, input_height), 
                                 input_details[0]['dtype'], use_rgb=False)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Get outputs (TFOD format: num_detections, boxes, scores, classes)
    outputs = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    
    if len(outputs) != 4:
        return []
    
    num_det, boxes, scores, classes = outputs
    
    # Flatten/squeeze dimensions
    num_detections = int(num_det.flatten()[0])
    boxes = boxes.squeeze()
    scores = scores.squeeze()
    classes = classes.squeeze()
    
    # Filter and convert
    detections = []
    mask = scores >= conf_threshold
    
    for i in np.where(mask)[0][:num_detections]:
        ymin, xmin, ymax, xmax = boxes[i]
        
        # Scale to image coordinates
        box_xyxy = np.array([xmin * w, ymin * h, xmax * w, ymax * h]).astype(int)
        box_xyxy = np.clip(box_xyxy, [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
        
        x1, y1, x2, y2 = box_xyxy
        if x2 > x1 and y2 > y1:
            detections.append(Detection(
                box=[x1, y1, x2, y2],
                confidence=float(scores[i]),
                class_id=int(classes[i])
            ))
    
    return detections


def detect_yolo_pt(model, img: np.ndarray, conf_threshold: float) -> List[Detection]:
    """YOLO PyTorch model detection"""
    results = model(img, conf=conf_threshold, verbose=False)[0]
    
    detections = []
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(Detection(
            box=xyxy.tolist(),
            confidence=float(box.conf[0]),
            class_id=int(box.cls[0])
        ))
    
    return detections


def visualize_detections(img: np.ndarray, detections: List[Detection], 
                         title: str, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw detections using cv2"""
    annotated = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.box
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add title
    cv2.putText(annotated, f"{title}: {len(detections)} detections", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return annotated


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("="*80)
    print("Triple Model Comparison Tool (Refactored)")
    print("="*80)
    
    # Load models
    models = {}
    
    # Current TFLite
    if CURRENT_TFLITE_PATH:
        try:
            interp = tf.lite.Interpreter(model_path=CURRENT_TFLITE_PATH)
            interp.allocate_tensors()
            models['current'] = {
                'interpreter': interp,
                'input_details': interp.get_input_details(),
                'output_details': interp.get_output_details(),
                'detect_fn': detect_yolo_tflite,
                'color': (0, 255, 0),
                'name': 'Current TFLite'
            }
            print(f"✓ Loaded Current TFLite")
        except Exception as e:
            print(f"⚠️  Could not load current TFLite: {e}")
    
    # Reference TFLite
    if REFERENCE_TFLITE_PATH:
        try:
            interp = tf.lite.Interpreter(model_path=REFERENCE_TFLITE_PATH)
            interp.allocate_tensors()
            models['reference'] = {
                'interpreter': interp,
                'input_details': interp.get_input_details(),
                'output_details': interp.get_output_details(),
                'detect_fn': detect_tfod_tflite,
                'color': (255, 0, 0),
                'name': 'Reference TFLite'
            }
            print(f"✓ Loaded Reference TFLite")
        except Exception as e:
            print(f"⚠️  Could not load reference TFLite: {e}")
    
    # YOLO PT
    if ORIGINAL_PT_MODEL and HAS_ULTRALYTICS:
        try:
            yolo_model = YOLO(ORIGINAL_PT_MODEL)
            models['yolo'] = {
                'model': yolo_model,
                'detect_fn': detect_yolo_pt,
                'color': (0, 165, 255),
                'name': 'YOLO PT'
            }
            print(f"✓ Loaded YOLO PT")
        except Exception as e:
            print(f"⚠️  Could not load YOLO: {e}")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    # Process images
    test_images = glob.glob(TEST_IMAGES_PATH)[:20]
    print(f"\nProcessing {len(test_images)} images...")
    
    # Statistics using pandas
    stats_data = []
    
    for img_idx, img_path in enumerate(test_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        images_to_stack = []
        
        for model_key, model_info in models.items():
            # Run detection
            if 'interpreter' in model_info:
                detections = model_info['detect_fn'](
                    model_info['interpreter'],
                    model_info['input_details'],
                    model_info['output_details'],
                    img, CONFIDENCE_THRESHOLD
                )
            else:
                detections = model_info['detect_fn'](model_info['model'], img, CONFIDENCE_THRESHOLD)
            
            # Apply NMS
            detections = apply_nms(detections)
            
            # Collect stats
            for det in detections:
                stats_data.append({
                    'model': model_info['name'],
                    'image': Path(img_path).name,
                    'class': det.class_name,
                    'confidence': det.confidence
                })
            
            # Visualize
            annotated = visualize_detections(img, detections, model_info['name'], model_info['color'])
            images_to_stack.append(cv2.resize(annotated, (640, int(h * 640 / w))))
        
        # Save comparison
        combined = np.hstack(images_to_stack)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"comparison_{Path(img_path).name}"), combined)
        
        if img_idx % 5 == 0:
            print(f"  Processed {img_idx + 1}/{len(test_images)} images...")
    
    # Print statistics using pandas
    print("\n[STATISTICS]")
    print("="*80)
    
    df = pd.DataFrame(stats_data)
    if not df.empty:
        summary = df.groupby(['model', 'class']).size().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        print(summary)
    
    print(f"\n✅ Results saved to: {OUTPUT_FOLDER}")
    print("="*80)


if __name__ == "__main__":
    main()