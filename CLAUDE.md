# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an inference package for eyelid/ocular segmentation using YOLOv11. It contains pre-trained model weights and works in conjunction with the main project at `C:\Users\CorneAI\YOLOv11_Detect_Seg_OBB`.

## Repository Contents

- `YOLO11n-detect.pt` - Stage1 model for bilateral eye detection from face images
- `YOLO11n-seg.pt` - Stage2 model for 6-class eye segmentation
- `RTDETR.pt` - Alternative detection model
- `venv/` - Python virtual environment

## Model Training Sources

### Stage1: YOLO11n-detect.pt
**Original model:** `C:\Users\CorneAI\FacePhoto_instance\models\295+cerebhq1-20000_yolo11l.pt`

**Training code:** `C:\Users\CorneAI\YOLOv11_Detect_Seg_OBB\train\train_stage1_bilateral.ipynb`

**Dataset:** `C:\Users\CorneAI\YOLOv11_Detect_Seg_OBB\Bilateral_data`
- Training: 4,286 images / Validation: 1,051 images
- Classes: `Right_eye` (0), `Left_eye` (1)
- Data sources: `1-295_cropped` + `Cereba-hq_1-9655`

**Training parameters:**
- Architecture: YOLOv11 Large
- Epochs: 200, Image size: 640, Batch: 16
- Optimizer: AdamW, Early stopping: 20

### Stage2: YOLO11n-seg.pt
**Original model:** `C:\Users\CorneAI\Eyelid_Iris_pupil_seg_comparison\YOLO11n-seg_3000mai\yolo11n-seg_3000mai\weights\best.pt`

**Training code:** `C:\Users\CorneAI\Eyelid_Iris_pupil_seg_comparison\train_yolo11n-seg_3000mai.ipynb`

**Dataset config:** `C:\Users\CorneAI\Eyelid_Iris_pupil_seg_comparison\YOLO11n-seg_3000mai\dataset_yolo11_3000mai.yaml`
- Training: 2,426 images (156 subjects) / Validation: 574 images (38 subjects)
- Total: 3,000 images

**6 Classes:**
- `conj` (0): Conjunctiva/Eyelid
- `caruncle` (1): Caruncle
- `iris_vis` (2): Visible iris
- `iris_occ` (3): Occluded iris
- `pupil_vis` (4): Visible pupil
- `pupil_occ` (5): Occluded pupil

**Training parameters:**
- Architecture: YOLOv11 Nano
- Epochs: 100, Image size: 512, Batch: 16
- Rotation: 0-180°, Mosaic: enabled

**Source annotations:**
- `Images\eyelid_caruncle_seg_0-3000.xml` (polygon)
- `Images\obb_iris_pupil_1-3000.xml` (ellipse)

## Related Project Structure

The main codebase is in `C:\Users\CorneAI\YOLOv11_Detect_Seg_OBB`:
- `infer/run_two_stage.py` - Two-stage inference pipeline (`TwoStageInference` class)
- `ops/roi_affine.py` - ROI extraction and coordinate transformation utilities (`ROIAffineTransform` class)
- `ops/cvat_to_yolo.py` - CVAT XML to YOLO format converter
- `train/` - Training notebooks for each stage

## Two-Stage Pipeline Architecture

1. **Stage1**: Face image → Bilateral eye BBox detection (Right/Left classification)
2. **Stage2**: Cropped ROI (512x512) → 6-class segmentation

### 6-Class Segmentation Output
- `conj` (0): Conjunctiva/Eyelid
- `caruncle` (1): Tear caruncle
- `iris_vis` (2): Visible iris portion
- `iris_occ` (3): Occluded iris (hidden by eyelid)
- `pupil_vis` (4): Visible pupil portion
- `pupil_occ` (5): Occluded pupil (hidden by eyelid)

## Commands

### Virtual Environment Activation
```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# Windows (Git Bash / MSYS2)
source ./venv/Scripts/activate

# Linux / Mac
source ./venv/bin/activate
```

### Install Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib pillow tqdm
```

### Run Inference
```python
from ultralytics import YOLO
import torch

# Load models
stage1 = YOLO("YOLO11n-detect.pt")
stage2 = YOLO("YOLO11n-seg.pt")

# Or use the TwoStageInference class from the main project
from infer.run_two_stage import TwoStageInference

pipeline = TwoStageInference(
    stage1_model_path="YOLO11n-detect.pt",
    stage2_model_path="YOLO11n-seg.pt",
    roi_size=512,
    expansion_ratio=0.25,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)
results = pipeline.infer("image.jpg", visualize=True)
```

### Verify GPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Ophthalmic Measurements

The system calculates (using 12mm corneal diameter calibration):
- **MRD-1**: Upper eyelid margin to pupil center distance
- **MRD-2**: Pupil center to lower eyelid margin distance
- **Palpebral Fissure Width**: Inner to outer canthus distance

## Key Implementation Details

- ROI extraction uses 25% expansion ratio and square padding
- Masks are transformed back to original image coordinates via `ROIAffineTransform`
- Confidence threshold default: 0.5
- Uses `retina_masks=True` for high-resolution segmentation output
