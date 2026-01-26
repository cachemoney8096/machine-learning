import tensorflow as tf
import numpy as np
import cv2

# Paths
SAVED_MODEL_PATH = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/coral_detection/coral-detection-model-v111/weights/best_saved_model"
TEST_IMAGE = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/calib_images/2025-Field-Tour-Video-Coral-Station_mp4-0010_jpg.rf.baddb14899d3a94757c1a37e393d9e61.jpg"  # Update with actual image

print("="*80)
print("DIAGNOSING RAW YOLO SAVEDMODEL OUTPUTS")
print("="*80)

# Load the SavedModel
print("\n[1] Loading SavedModel...")
model = tf.saved_model.load(SAVED_MODEL_PATH)
infer = model.signatures["serving_default"]

print("Model loaded successfully!")
print(f"Input signature: {infer.structured_input_signature}")
print(f"Output signature: {infer.structured_outputs}")

# Load and preprocess test image
print(f"\n[2] Loading test image: {TEST_IMAGE}")
img = cv2.imread(TEST_IMAGE)
img = cv2.resize(img, (640, 640))
img_float = img.astype(np.float32) / 255.0
img_batch = np.expand_dims(img_float, axis=0)

print(f"Image shape: {img_batch.shape}")
print(f"Image dtype: {img_batch.dtype}")
print(f"Image range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")

# Run inference
print("\n[3] Running inference...")
outputs = infer(tf.constant(img_batch))

print(f"Number of outputs: {len(outputs)}")
for key, value in outputs.items():
    print(f"\nOutput '{key}':")
    print(f"  Shape: {value.shape}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Range: [{value.numpy().min():.6f}, {value.numpy().max():.6f}]")
    print(f"  First few values: {value.numpy().flatten()[:10]}")

# Get the main output (should be [1, 6, 8400] for YOLOv8)
preds = list(outputs.values())[0].numpy()
print(f"\n[4] Analyzing predictions shape: {preds.shape}")

if len(preds.shape) == 3:
    # Expected format: [batch, features, anchors]
    # Features: [x, y, w, h, objectness, class_scores...]
    num_features = preds.shape[1]
    num_anchors = preds.shape[2]
    
    print(f"Number of features: {num_features}")
    print(f"Number of anchors: {num_anchors}")
    
    # Split into components
    boxes = preds[0, 0:4, :]  # x, y, w, h
    objectness = preds[0, 4:5, :]
    class_scores = preds[0, 5:, :] if num_features > 5 else None
    
    print(f"\n[5] Box coordinates (first 5 anchors):")
    print(f"  X range: [{boxes[0, :].min():.2f}, {boxes[0, :].max():.2f}]")
    print(f"  Y range: [{boxes[1, :].min():.2f}, {boxes[1, :].max():.2f}]")
    print(f"  W range: [{boxes[2, :].min():.2f}, {boxes[2, :].max():.2f}]")
    print(f"  H range: [{boxes[3, :].min():.2f}, {boxes[3, :].max():.2f}]")
    print(f"  First 3 boxes (xywh):")
    for i in range(min(3, num_anchors)):
        print(f"    Anchor {i}: x={boxes[0,i]:.2f}, y={boxes[1,i]:.2f}, w={boxes[2,i]:.2f}, h={boxes[3,i]:.2f}")
    
    print(f"\n[6] Objectness scores:")
    print(f"  Range: [{objectness.min():.6f}, {objectness.max():.6f}]")
    print(f"  Mean: {objectness.mean():.6f}")
    print(f"  Scores > 0.5: {(objectness > 0.5).sum()}")
    print(f"  Scores > 0.25: {(objectness > 0.25).sum()}")
    print(f"  Top 10 scores: {np.sort(objectness.flatten())[-10:]}")
    
    if class_scores is not None:
        print(f"\n[7] Class scores:")
        print(f"  Shape: {class_scores.shape}")
        print(f"  Range: [{class_scores.min():.6f}, {class_scores.max():.6f}]")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if objectness.max() < 0.01:
    print("❌ PROBLEM: Objectness scores are near zero!")
    print("   The model isn't detecting anything even in raw form.")
    print("   This suggests an issue with the SavedModel export itself.")
elif objectness.max() > 0.25:
    print("✅ Raw model IS detecting objects!")
    print(f"   Max objectness: {objectness.max():.3f}")
    print(f"   Detections above 0.25: {(objectness > 0.25).sum()}")
    print("\n   → The problem is in the NMS wrapper, not the base model!")
    print("   → Check coordinate transformation in the NMS code")
else:
    print("⚠️  Objectness scores are low but not zero")
    print(f"   Max: {objectness.max():.3f}")
    print("   This might work with lower thresholds")

print("\n" + "="*80)