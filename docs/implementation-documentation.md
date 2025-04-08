# Web Segmentation Research Implementation Documentation

This document provides a comprehensive overview of the implementation details for the web segmentation research project, focusing on the extensions of existing models for web page segmentation tasks.

## Project Overview

This research project extends existing computer vision models to address the challenges of web page segmentation, with a focus on:

1. **YOLO-WS**: An extension of YOLOv5 for web segmentation
2. **WEB-SAM**: An adaptation of Facebook's Segment Anything Model (SAM) for web content
3. **Comparative Analysis**: Benchmarking against YOLOv5 and YOLOv11 on web segmentation datasets

## Repository Structure

```
vt2-visual-webseg/
├── src/
│   ├── yolov5/                  # YOLO-WS implementation
│   │   ├── models/              # Modified model architectures
│   │   │   ├── yolov5sWS.yaml   # Small YOLO-WS model config
│   │   │   ├── yolov5mWS.yaml   # Medium YOLO-WS model config
│   │   │   ├── yolov5lWS.yaml   # Large YOLO-WS model config
│   │   │   ├── yolov5nWS.yaml   # Nano YOLO-WS model config
│   │   │   └── yolov5xWS.yaml   # XLarge YOLO-WS model config
│   │   ├── train_yolowsv5.py    # Custom training script for YOLO-WS
│   │   └── val_wbf.py           # Validation with Weighted Boxes Fusion
│   ├── websam/                  # WEB-SAM implementation
│   │   ├── train_websam.py      # Training script for WEB-SAM
│   │   └── train-websam.ipynb   # Notebook for WEB-SAM training
│   └── bcubed_f1_websam.py      # B-Cubed F1 metric implementation
├── docs/
│   ├── yolov5-ws-modifications.md  # Detailed YOLO-WS modifications
│   └── websam-modifications.md     # Detailed WEB-SAM modifications
└── evaluation/                  # Evaluation results and visualizations
```

## Base Implementations

### 1. YOLOv5 Base

- **Original Repository**: Ultralytics YOLOv5
- **URL**: https://github.com/ultralytics/yolov5
- **Commit ID**: e62a31b6 (December 7, 2024)

The YOLO-WS implementation in `/src/yolov5/` is based on this specific commit of the original repository. For detailed documentation of modifications, see [YOLO-WS Modifications](./yolov5-ws-modifications.md).

### 2. Segment Anything Model (SAM) Base

- **Original Repository**: Facebook Research's Segment Anything Model
- **URL**: https://github.com/facebookresearch/segment-anything.git
- **Commit ID**: dca509fe793f601edb92606367a655c15ac00fdf (September 18, 2024)

The WEB-SAM implementation in `/src/websam/` is based on this specific commit of the original repository. For detailed documentation of modifications, see [WEB-SAM Modifications](./websam-modifications.md).

## Key Modifications Summary

### YOLO-WS Implementation

The YOLO-WS implementation extends YOLOv5 with the following key modifications:

1. **Model Architecture Changes**:
   - **`src/yolov5/models/*.yaml`**: Modified model configurations with specific depth and width multipliers
   - **`src/yolov5/models/common.py`**: Implemented C3WS modules that replace standard C3 modules for weakly supervised learning
   - Modified ConvFrelu activation functions for better feature extraction on web content
   - Customized anchor configurations specifically optimized for web UI elements

2. **Training Pipeline Modifications**:
   - **`src/yolov5/train_yolowsv5.py`**: Main training script that initializes the `CustomComputeLoss` class (line 370)
   - **`src/yolov5/custom/loss.py`**: Implemented custom loss function with:
     - `CustomComputeLoss` class (line 90) that combines EIoU loss and focal loss
     - `bbox_eiou` function (line 37) for improved bounding box regression
     - Focal loss implementation (lines 226-234) with gamma parameter for better handling of hard examples
   - **`src/yolov5/utils/datasets.py`**: **Critical Improvement**: Implemented proper handling of nested multipolygon structures in version 3, which significantly improved performance compared to the original paper's results

3. **Evaluation Enhancements**:
   - **`src/yolov5/val_wbf.py`**: Implemented Weighted Boxes Fusion (WBF) instead of standard NMS for better handling of overlapping web elements
   - **`src/yolov5/utils/metrics.py`**: Custom evaluation metrics specifically optimized for web segmentation tasks
   - Specialized processing for web UI element detection with improved precision and recall

For detailed information on the YOLO-WS implementation, including specific file modifications and technical approaches, see [YOLO-WS Modifications](./yolov5-ws-modifications.md).

### WEB-SAM Implementation

The WEB-SAM implementation extends SAM with these key modifications:

1. **Training Capability Enhancement**:
   - **`src/websam/segment_anything/modeling/sam.py`**: Removed `@torch.no_grad()` decorator (line 52) to enable gradient computation
   - **`src/websam/segment_anything/build_sam.py`**: Added parameters for training control with `strict_weights` and `freeze_encoder` options (lines 11-36)
   - **`src/websam/train_websam.py`**: Implemented custom loss function with:
     - `BCEIoULoss` class (lines 141-160) that combines Binary Cross Entropy and IoU loss
     - Mixed precision training using `torch.cuda.amp.GradScaler` (line 23)
   - Added support for proper multipolygon handling with hole recognition in the data loading pipeline

2. **Architecture Enhancements**:
   - **`src/websam/segment_anything/modeling/image_encoder.py`**: Added specialized components:
     - `PatchEmbeddingTune` class (lines 432-438) with scale factor of 8
     - `EdgeComponentsTune` class (lines 440-518) with 16×16 kernel for better edge detection
     - `Adapter` class (lines 520-554) for efficient fine-tuning

3. **Evaluation Implementation**:
   - **`src/bcubed_f1_websam.py`**: Implementation of the B-Cubed F1 metric for evaluating segmentation quality
   - **`src/websam/train_websam.py`**: `calculate_metrics` function (lines 202-237) for computing PB3, RB3, F*B3, F1, and Pixel Accuracy

For detailed information on the WEB-SAM implementation, including specific file modifications and architectural enhancements, see [WEB-SAM Modifications](./websam-modifications.md).

## Datasets

The implementations were trained and evaluated on:

1. **Webis-Webseg-20**: The primary dataset for web segmentation
2. **Nano**: A smaller custom dataset for specialized testing

## Research Findings

Key findings from the research include:

1. **Improved Implementation**: Fixed implementation issues in YOLO-WS to achieve better results compared to the original paper
2. **Nested Structures**: Better handling of nested multipolygon structures significantly improved performance
3. **Instance vs. Object Detection**: Instance segmentation didn't substantially outperform object detection
4. **Training Set Size**: Performance improvements followed a logarithmic pattern with respect to training set size

## Future Work

Planned extensions of this research include:

1. Additional experiments with WEB-SAM
2. Exploration of webpage splitting techniques
3. Further optimization of the weak supervision approach

## Repository Structure

- `/src/yolov5/`: YOLO-WS implementation
- `/src/websam/`: WEB-SAM implementation
- `/docs/`: Documentation and research papers
- `/models/`: Model outputs and weights
- `/evaluation/`: Evaluation scripts and results

## References

1. Ultralytics YOLOv5: https://github.com/ultralytics/yolov5
2. Facebook Research's Segment Anything Model: https://github.com/facebookresearch/segment-anything
3. Webis-Webseg-20 Dataset: https://webis.de/data/webis-webseg-20.html
