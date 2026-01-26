import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION - Update these paths
# ============================================================================
TFLITE_MODEL_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/limelight_ready_5.tflite"
TEST_IMAGES_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/calib_images/*"
OUTPUT_FOLDER = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/detection_results"
ORIGINAL_PT_MODEL = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best.pt"

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to show detection
IOU_THRESHOLD = 0.5

# ============================================================================
# SETUP
# ============================================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print("="*80)
print("TFLite Model Debugging Tool")
print("="*80)

# ============================================================================
# PART 1: Inspect TFLite Model
# ============================================================================
print("\n[1] INSPECTING TFLITE MODEL")
print("-"*80)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput Details:")
for i, detail in enumerate(input_details):
    print(f"  Input {i}:")
    print(f"    Name: {detail['name']}")
    print(f"    Shape: {detail['shape']}")
    print(f"    Type: {detail['dtype']}")
    print(f"    Quantization: scale={detail['quantization'][0]}, zero_point={detail['quantization'][1]}")

print("\nOutput Details:")
for i, detail in enumerate(output_details):
    print(f"  Output {i}:")
    print(f"    Name: {detail['name']}")
    print(f"    Shape: {detail['shape']}")
    print(f"    Type: {detail['dtype']}")
    print(f"    Quantization: {detail['quantization']}")

# Get model input requirements
input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]
input_dtype = input_details[0]['dtype']

print(f"\nModel expects: {input_dtype.__name__} images of size [{input_height}, {input_width}]")

# ============================================================================
# PART 2: Test TFLite Model on Images
# ============================================================================
print("\n[2] RUNNING TFLITE DETECTIONS")
print("-"*80)

test_images = glob.glob(TEST_IMAGES_PATH)[:20]  # Test on first 20 images
print(f"Testing on {len(test_images)} images...")

detection_stats = {
    'total_images': 0,
    'images_with_detections': 0,
    'total_detections': 0,
    'confidence_scores': []
}

for img_idx, img_path in enumerate(test_images):
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠️  Could not load {img_path}")
        continue
    
    original_img = img.copy()
    h, w = img.shape[:2]
    
    # Resize to model input size
    img_resized = cv2.resize(img, (input_width, input_height))
    
    # Prepare input based on model's expected dtype
    if input_dtype == np.uint8:
        input_data = img_resized.astype(np.uint8)
    else:
        input_data = img_resized.astype(np.float32) / 255.0
    
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs
    # Based on TF Object Detection API format:
    # Output 0: scores [1, 10]
    # Output 1: boxes [1, 10, 4]
    # Output 2: num_detections [1]
    # Output 3: classes [1, 10]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # [10]
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]   # [10, 4]
    num_detections = int(interpreter.get_tensor(output_details[2]['index'])[0])
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # [10]
    
    # Debug: Print raw outputs for first image
    if img_idx == 0:
        print(f"\n  📊 Raw outputs for first image:")
        print(f"     num_detections: {num_detections}")
        print(f"     scores (first 5): {scores[:5]}")
        print(f"     boxes (first 2): {boxes[:2]}")
        print(f"     Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"     Box coord range: [{boxes.min():.4f}, {boxes.max():.4f}]")
    
    detection_stats['total_images'] += 1
    
    # Filter detections by confidence
    valid_detections = 0
    for i in range(len(scores)):
        if scores[i] >= CONFIDENCE_THRESHOLD:
            valid_detections += 1
            detection_stats['total_detections'] += 1
            detection_stats['confidence_scores'].append(float(scores[i]))
            
            # Draw detection on image
            box = boxes[i]
            # Boxes are in normalized [y1, x1, y2, x2] format
            y1, x1, y2, x2 = box
            
            # Convert to pixel coordinates
            x1_px = int(x1 * w)
            y1_px = int(y1 * h)
            x2_px = int(x2 * w)
            y2_px = int(y2 * h)
            
            # Draw bounding box
            cv2.rectangle(original_img, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
            
            # Add label
            label = f"Coral {scores[i]:.2f}"
            cv2.putText(original_img, label, (x1_px, y1_px - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if valid_detections > 0:
        detection_stats['images_with_detections'] += 1
    
    # Add summary text to image
    summary = f"TFLite: {valid_detections} detections | Max conf: {scores.max():.3f}"
    cv2.putText(original_img, summary, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save result
    output_path = os.path.join(OUTPUT_FOLDER, f"tflite_{Path(img_path).name}")
    cv2.imwrite(output_path, original_img)
    
    if img_idx % 5 == 0:
        print(f"  Processed {img_idx + 1}/{len(test_images)} images...")

print(f"\n✅ TFLite testing complete!")
print(f"   Results saved to: {OUTPUT_FOLDER}")

# ============================================================================
# PART 3: Test Original YOLO Model for Comparison
# ============================================================================
print("\n[3] TESTING ORIGINAL YOLO MODEL (FOR COMPARISON)")
print("-"*80)

try:
    from ultralytics import YOLO
    
    yolo_model = YOLO(ORIGINAL_PT_MODEL)
    print(f"Loaded YOLO model from {ORIGINAL_PT_MODEL}")
    
    yolo_stats = {
        'total_images': 0,
        'images_with_detections': 0,
        'total_detections': 0,
        'confidence_scores': []
    }
    
    for img_idx, img_path in enumerate(test_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Run YOLO inference
        results = yolo_model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        
        yolo_stats['total_images'] += 1
        
        # Draw detections
        original_img = img.copy()
        num_dets = len(results.boxes)
        
        if num_dets > 0:
            yolo_stats['images_with_detections'] += 1
        
        for box in results.boxes:
            yolo_stats['total_detections'] += 1
            conf = float(box.conf[0])
            yolo_stats['confidence_scores'].append(conf)
            
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Draw
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Coral {conf:.2f}"
            cv2.putText(original_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add summary
        max_conf = max([float(b.conf[0]) for b in results.boxes]) if num_dets > 0 else 0
        summary = f"YOLO: {num_dets} detections | Max conf: {max_conf:.3f}"
        cv2.putText(original_img, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save
        output_path = os.path.join(OUTPUT_FOLDER, f"yolo_{Path(img_path).name}")
        cv2.imwrite(output_path, original_img)
    
    print(f"✅ YOLO testing complete!")
    
except Exception as e:
    print(f"⚠️  Could not test YOLO model: {e}")
    yolo_stats = None

# ============================================================================
# PART 4: Statistical Analysis
# ============================================================================
print("\n[4] STATISTICAL ANALYSIS")
print("="*80)

print("\n📊 TFLite Model Statistics:")
print(f"   Total images tested: {detection_stats['total_images']}")
print(f"   Images with detections: {detection_stats['images_with_detections']}")
print(f"   Total detections: {detection_stats['total_detections']}")
if detection_stats['confidence_scores']:
    print(f"   Avg confidence: {np.mean(detection_stats['confidence_scores']):.3f}")
    print(f"   Max confidence: {np.max(detection_stats['confidence_scores']):.3f}")
    print(f"   Min confidence: {np.min(detection_stats['confidence_scores']):.3f}")
else:
    print(f"   ⚠️  NO DETECTIONS FOUND!")

if yolo_stats:
    print(f"\n📊 Original YOLO Model Statistics:")
    print(f"   Total images tested: {yolo_stats['total_images']}")
    print(f"   Images with detections: {yolo_stats['images_with_detections']}")
    print(f"   Total detections: {yolo_stats['total_detections']}")
    if yolo_stats['confidence_scores']:
        print(f"   Avg confidence: {np.mean(yolo_stats['confidence_scores']):.3f}")
        print(f"   Max confidence: {np.max(yolo_stats['confidence_scores']):.3f}")
        print(f"   Min confidence: {np.min(yolo_stats['confidence_scores']):.3f}")

# ============================================================================
# PART 5: Diagnosis
# ============================================================================
print("\n[5] DIAGNOSIS")
print("="*80)

if detection_stats['total_detections'] == 0:
    print("\n⚠️  PROBLEM: TFLite model produced ZERO detections!")
    print("\nPossible causes:")
    print("  1. ❌ Coordinate format mismatch (normalized vs pixel)")
    print("  2. ❌ Score threshold too high")
    print("  3. ❌ Input preprocessing incorrect")
    print("  4. ❌ Quantization degraded model quality")
    print("  5. ❌ NMS parameters too strict")
    
    if yolo_stats and yolo_stats['total_detections'] > 0:
        print(f"\n✅ Original YOLO found {yolo_stats['total_detections']} detections")
        print("   → This confirms the conversion process lost accuracy!")
    
    print("\n🔍 Next steps:")
    print("  - Check the saved images in the output folder")
    print("  - Look at the raw score values printed above")
    print("  - Try lowering CONFIDENCE_THRESHOLD to 0.01 and re-run")
    print("  - Check if boxes have valid coordinates (0-1 range)")

elif yolo_stats and detection_stats['total_detections'] < yolo_stats['total_detections'] * 0.5:
    print(f"\n⚠️  WARNING: TFLite found {detection_stats['total_detections']} detections")
    print(f"   vs YOLO found {yolo_stats['total_detections']} detections")
    print("   → Significant accuracy loss during conversion!")
    
else:
    print(f"\n✅ TFLite model is working! Found {detection_stats['total_detections']} detections")
    if yolo_stats:
        print(f"   (YOLO found {yolo_stats['total_detections']} for comparison)")

print(f"\n📁 All results saved to: {OUTPUT_FOLDER}")
print("   Review the images to see what the model is detecting!")
print("="*80)