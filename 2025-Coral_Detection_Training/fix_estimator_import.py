import os
import re

# Path to the file that needs fixing
file_path = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/models/research/object_detection/inputs.py'

with open(file_path, 'r') as f:
    content = f.read()

# Replace the problematic import
old_import = "from tensorflow.compat.v1 import estimator as tf_estimator"
new_import = "import tensorflow_estimator as tf_estimator"

if old_import in content:
    content = content.replace(old_import, new_import)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"✓ Fixed estimator import in {file_path}")
else:
    print("Import already fixed or not found")
