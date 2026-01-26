import tensorflow as tf
import numpy as np

def describe_model(path):
    print(f"\n{'='*80}")
    print(f"MODEL: {path}")
    print(f"{'='*80}")

    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    details = {
        "inputs": interpreter.get_input_details(),
        "outputs": interpreter.get_output_details(),
        "tensors": interpreter.get_tensor_details(),
    }

    # ---- Inputs ----
    print("\n[INPUTS]")
    for i, d in enumerate(details["inputs"]):
        print(f"Input {i}")
        print(f"  name: {d['name']}")
        print(f"  shape: {d['shape']}")
        print(f"  dtype: {d['dtype']}")
        print(f"  quantization: {d['quantization']}")
        print(f"  quantization_parameters: {d['quantization_parameters']}")

    # ---- Outputs ----
    print("\n[OUTPUTS]")
    for i, d in enumerate(details["outputs"]):
        print(f"Output {i}")
        print(f"  name: {d['name']}")
        print(f"  shape: {d['shape']}")
        print(f"  dtype: {d['dtype']}")
        print(f"  quantization: {d['quantization']}")
        print(f"  quantization_parameters: {d['quantization_parameters']}")

    # ---- Tensor stats ----
    dtypes = {}
    for t in details["tensors"]:
        dtypes[t["dtype"]] = dtypes.get(t["dtype"], 0) + 1

    print("\n[TENSOR DTYPE COUNTS]")
    for k, v in dtypes.items():
        print(f"  {k}: {v}")

    # ---- Quantized vs float ----
    quantized = sum(
        1 for t in details["tensors"]
        if t["quantization_parameters"]["scales"].size > 0
    )

    print(f"\n[QUANTIZED TENSORS]: {quantized} / {len(details['tensors'])}")

    # ---- Ops ----
    ops = set()
    for t in details["tensors"]:
        ops.add(t["name"].split('/')[-1])

    print(f"\n[UNIQUE TENSOR NAMES] ({len(ops)})")
    for op in sorted(list(ops))[:30]:
        print(f"  {op}")
    if len(ops) > 30:
        print("  ...")

    return details


# ======== COMPARE TWO MODELS ========
model_a = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/B2_CPU_coral_and_algae_monochrome.tflite"
model_b = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2025-Coral_Detection_Training/limelight_ready_3.tflite"

a = describe_model(model_a)
b = describe_model(model_b)

print("\n" + "="*80)
print("DIFFERENCE SUMMARY")
print("="*80)

def diff_inputs(a, b):
    print("\n[INPUT DIFF]")
    for i in range(min(len(a["inputs"]), len(b["inputs"]))):
        da, db = a["inputs"][i], b["inputs"][i]
        if da["dtype"] != db["dtype"]:
            print(f"Input {i} dtype differs: {da['dtype']} vs {db['dtype']}")
        if not np.array_equal(da["shape"], db["shape"]):
            print(f"Input {i} shape differs: {da['shape']} vs {db['shape']}")
        if da["quantization"] != db["quantization"]:
            print(f"Input {i} quant differs: {da['quantization']} vs {db['quantization']}")

def diff_outputs(a, b):
    print("\n[OUTPUT DIFF]")
    for i in range(min(len(a["outputs"]), len(b["outputs"]))):
        da, db = a["outputs"][i], b["outputs"][i]
        if da["dtype"] != db["dtype"]:
            print(f"Output {i} dtype differs: {da['dtype']} vs {db['dtype']}")
        if not np.array_equal(da["shape"], db["shape"]):
            print(f"Output {i} shape differs: {da['shape']} vs {db['shape']}")
        if da["quantization"] != db["quantization"]:
            print(f"Output {i} quant differs: {da['quantization']} vs {db['quantization']}")

diff_inputs(a, b)
diff_outputs(a, b)
