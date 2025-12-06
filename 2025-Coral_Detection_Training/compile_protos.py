#!/usr/bin/env python3
"""
Compile protobuf files using Python's grpc_tools
This avoids version mismatches with system protoc
"""

import os
import sys
import glob

BASE_DIR = '/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training'

print("=== Installing grpcio-tools ===")
os.system('pip install -q protobuf==3.20.3 grpcio-tools')

print("\n=== Compiling protobuf files ===")

try:
    from grpc_tools import protoc as grpc_protoc
except ImportError:
    print("ERROR: grpcio-tools not installed")
    print("Run: pip install grpcio-tools")
    sys.exit(1)

research_dir = os.path.join(BASE_DIR, 'models', 'research')
proto_dir = os.path.join(research_dir, 'object_detection/protos')

# Clean old compiled files
print("Cleaning old compiled proto files...")
for pb2_file in glob.glob(os.path.join(proto_dir, '*_pb2.py')):
    os.remove(pb2_file)
    print(f"  Removed {os.path.basename(pb2_file)}")

# Find all proto files
proto_files = glob.glob(os.path.join(proto_dir, '*.proto'))
if not proto_files:
    print(f"ERROR: No .proto files found in {proto_dir}")
    sys.exit(1)

print(f"\nCompiling {len(proto_files)} proto files using Python's protoc...")

failed = []
succeeded = []

for proto_file in proto_files:
    proto_name = os.path.basename(proto_file)
    
    # Use grpc_tools.protoc
    result = grpc_protoc.main([
        'grpc_tools.protoc',
        f'-I{research_dir}',
        f'--python_out={research_dir}',
        f'object_detection/protos/{proto_name}'
    ])
    
    if result != 0:
        failed.append(proto_name)
        print(f"✗ {proto_name}")
    else:
        succeeded.append(proto_name)
        print(f"✓ {proto_name}")

print(f"\n=== Results ===")
print(f"Succeeded: {len(succeeded)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed files:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\n✓ All proto files compiled successfully!")
    print("\nYou can now run: python train_model.py")