import tensorflow as tf
import numpy as np
import cv2
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================
SAVED_MODEL_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/tf_model_nms"
TEST_IMAGES_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/calib_images/*"

print("="*80)
print("SAVEDMODEL NMS WRAPPER TEST")
print("="*80)

# Load the SavedModel
print(f"\nLoading SavedModel from: {SAVED_MODEL_PATH}")
try:
    model = tf.saved_model.load(SAVED_MODEL_PATH)
    print("✅ SavedModel loaded successfully")
except Exception as e:
    print(f"❌ Failed to load SavedModel: {e}")
    exit(1)

# Get the inference function
try:
    infer = model.signatures["serving_default"]
    print("✅ Found serving_default signature")
except:
    print("❌ No serving_default signature found")
    print("Available signatures:", list(model.signatures.keys()))
    exit(1)

# Load test images
test_images = glob.glob(TEST_IMAGES_PATH)[:5]
print(f"\nTesting on {len(test_images)} images...")

for idx, img_path in enumerate(test_images):
    print(f"\n{'='*80}")
    print(f"Image {idx+1}: {img_path}")
    print('='*80)
    
    # Load and preprocess
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not load image")
        continue
    
    img_resized = cv2.resize(img, (640, 640))
    img_float = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_float, axis=0)
    
    # Run inference
    try:
        outputs = infer(tf.constant(img_batch))
        print("✅ Inference successful")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # Print all outputs
    print(f"\nOutputs ({len(outputs)} total):")
    for key, value in outputs.items():
        arr = value.numpy()
        print(f"\n  {key}:")
        print(f"    shape: {arr.shape}")
        print(f"    dtype: {arr.dtype}")
        print(f"    range: [{arr.min():.6f}, {arr.max():.6f}]")
        
        # Print specific values for each output type
        if arr.size == 1:
            print(f"    value: {arr.flatten()[0]}")
        elif len(arr.shape) == 2 and arr.shape[1] == 10:
            scores = arr[0]
            print(f"    first 5 values: {scores[:5]}")
            print(f"    max value: {scores.max():.6f}")
            print(f"    values > 0.5: {np.sum(scores > 0.5)}")
            print(f"    values > 0.25: {np.sum(scores > 0.25)}")
            print(f"    values > 0.01: {np.sum(scores > 0.01)}")
        elif len(arr.shape) == 3 and arr.shape[2] == 4:
            boxes = arr[0]
            print(f"    first box: {boxes[0]}")
    
    # Try to identify and extract key values
    output_list = list(outputs.values())
    
    # Find num_detections
    num_dets = None
    for out in output_list:
        if out.numpy().size == 1:
            num_dets = int(out.numpy().flatten()[0])
            print(f"\n🎯 Found num_detections: {num_dets}")
            break
    
    # Find scores
    for out in output_list:
        arr = out.numpy()
        if len(arr.shape) == 2 and arr.shape[1] == 10:
            scores = arr[0]
            print(f"🎯 Detection scores:")
            print(f"    Max score: {scores.max():.6f}")
            if scores.max() > 0:
                print(f"    Top 3 scores: {sorted(scores, reverse=True)[:3]}")
            break
    
    # Summary
    if num_dets == 0:
        print("\n⚠️  NO DETECTIONS on this image!")
    elif num_dets > 0:
        print(f"\n✅ Found {num_dets} detections!")

print("\n" + "="*80)
print("SAVEDMODEL TEST COMPLETE")
print("="*80)

# ============================================================================
# Additional debugging: Test raw YOLO output
# ============================================================================
print("\n\nDEBUG: Testing raw YOLO output for comparison...")

from ultralytics import YOLO

PT_MODEL_PATH = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best.pt'
yolo = YOLO(PT_MODEL_PATH)

img = cv2.imread(test_images[0])
img_resized = cv2.resize(img, (640, 640))

# Get YOLO detections
results = yolo.predict(img_resized, verbose=False)[0]
print(f"\nYOLO found {len(results.boxes)} detections")
if len(results.boxes) > 0:
    confs = [float(b.conf[0]) for b in results.boxes]
    print(f"Confidences: {confs}")

# Get raw YOLO outputs
print("\nComparing with raw exported YOLO model...")
SAVED_MODEL_RAW = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best_saved_model"
raw_model = tf.saved_model.load(SAVED_MODEL_RAW)
raw_infer = raw_model.signatures["serving_default"]

img_float = img_resized.astype(np.float32) / 255.0
img_batch = np.expand_dims(img_float, axis=0)

raw_out = raw_infer(tf.constant(img_batch))
preds = list(raw_out.values())[0].numpy()  # [1, 6, 8400]

print(f"\nRaw YOLO output shape: {preds.shape}")
print(f"Raw output range: [{preds.min():.2f}, {preds.max():.2f}]")

# Look at the objectness scores
objectness = preds[0, 4, :]  # [8400]
print(f"\nRaw objectness stats:")
print(f"  Range: [{objectness.min():.2f}, {objectness.max():.2f}]")
print(f"  Mean: {objectness.mean():.2f}")
print(f"  Top 5 values: {sorted(objectness, reverse=True)[:5]}")

# Apply sigmoid to see what we should get
objectness_sigmoid = 1 / (1 + np.exp(-objectness))
print(f"\nAfter sigmoid:")
print(f"  Range: [{objectness_sigmoid.min():.6f}, {objectness_sigmoid.max():.6f}]")
print(f"  Mean: {objectness_sigmoid.mean():.6f}")
print(f"  Top 5 values: {sorted(objectness_sigmoid, reverse=True)[:5]}")
print(f"  Values > 0.25: {np.sum(objectness_sigmoid > 0.25)}")

print("\n" + "="*80)
print("If raw objectness values are LARGE (>10), they need sigmoid")
print("If raw objectness values are small (0-1), they're already sigmoid'd")
print("="*80)