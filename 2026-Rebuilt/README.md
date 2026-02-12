# 2026 Rebuilt: Dataset to RKNN

This README summarizes the steps from `robot_detection` dataset to a final `.rknn` model, based on `yolo_training.ipynb` and `rknn_conversion.ipynb`.

## 0) Prereqs

- Python 3.9+ and pip
- Git (required for ONNX export helper)
- Linux for RKNN conversion (`rknn-toolkit2` is Linux-only; use Colab if needed)

Optional but recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If you plan to run the notebooks, also install Jupyter:

```bash
python -m pip install jupyter
```

## 1) Install training deps

```bash
python -m pip install --upgrade ultralytics
```

## 2) Train YOLO on the dataset in `robot_detection`

Open `2026-Rebuilt/yolo_training.ipynb` and edit the first two cells.

Set your run mode and path:

```python
CLUSTER = False  # set True if running on your cluster
HOME = "/Users/rubenhayrapetyan/Downloads/Code/FRC/machine-learning/2026-Rebuilt"  # update to your path
```

Then update the training cell so the dataset points at `robot_detection`:

```python
results = model.train(
    data=f"{HOME}/robot_detection/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="robot-detection-model-v1",
    workers=8,
    project=f"{HOME}/robot_detection/runs/train",
)
```

Notes:
- Leave `CLUSTER = True` only if you are using the cluster branch and want CUDA device selection.
- `yolov8s.pt` will be downloaded on first run.

After training, your weights are here:

```
2026-Rebuilt/robot_detection/runs/train/robot-detection-model-v1/weights/best.pt
```

## 3) Export to ONNX (Linux/Colab)

Open `2026-Rebuilt/rknn_conversion.ipynb` and run the setup cell labeled **Create ONNX/RKNN**.

Then run:

```python
create_onnx(
    model_path="2026-Rebuilt/robot_detection/runs/train/robot-detection-model-v1/weights/best.pt",
    version="yolov8",
)
```

This will generate an ONNX model (usually next to the `.pt` file).

## 4) Install RKNN Toolkit 2 (Linux only)

```bash
python -m pip install rknn-toolkit2
```

If the automatic install fails, install the matching `.whl` from:
`https://github.com/airockchip/rknn-toolkit2`

## 5) Convert ONNX to RKNN

Use `robot_detection` as the calibration image source (it must contain `data.yaml` or a folder of images):

```python
create_rknn(
    img_dir="2026-Rebuilt/robot_detection",
    model_path="2026-Rebuilt/robot_detection/runs/train/robot-detection-model-v1/weights/best.onnx",
    rknn_output="2026-Rebuilt/robot_detection/runs/train/robot-detection-model-v1/weights/robot-detection-model-v1.rknn",
    num_imgs=300,
    disable_quantize=False,
)
```

Notes:
- Leave `disable_quantize=False` (quantization is required for PhotonVision).
- If your dataset has fewer than 300 images, lower `num_imgs`.
- The quantization image list is saved to `imgs.txt` in your working directory.

## 6) Output

Your final RKNN model will be at:

```
2026-Rebuilt/robot_detection/runs/train/robot-detection-model-v1/weights/robot-detection-model-v1.rknn
```
