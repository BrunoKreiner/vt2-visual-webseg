# YOLO-WS: Modifications to YOLOv5 for Web Segmentation

This document details the modifications made to the original YOLOv5 model to create YOLO-WS for web page segmentation tasks.

## Base Repository Information

- **Original Repository**: Ultralytics YOLOv5
- **URL**: https://github.com/ultralytics/yolov5
- **Commit ID**: e62a31b6 (December 7, 2024)
- **Commit Message**: "Update format.yml (#13451)"
- **License**: AGPL-3.0

The implementation in `/src/yolov5/` is based on this specific commit of the original repository. For reference, the original implementation can be accessed directly via:
```
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout e62a31b6
```

## Implementation Overview

The YOLO-WS implementation extends the YOLOv5 architecture to better handle web page segmentation tasks. The implementation spans several key files, with the primary modifications in the model architecture files and training scripts.

## Modified Files

The following key files were modified or created to implement YOLO-WS:

### 1. Model Architecture Files

- **`src/yolov5/models/yolov5sWS.yaml`**: Small YOLO-WS model configuration
- **`src/yolov5/models/yolov5mWS.yaml`**: Medium YOLO-WS model configuration
- **`src/yolov5/models/yolov5lWS.yaml`**: Large YOLO-WS model configuration
- **`src/yolov5/models/yolov5nWS.yaml`**: Nano YOLO-WS model configuration
- **`src/yolov5/models/yolov5xWS.yaml`**: XLarge YOLO-WS model configuration
- **`src/yolov5/models/common.py`**: Added C3WS module for weakly supervised learning with specialized convolutional layers

### 2. Training Implementation

- **`src/yolov5/train_yolowsv5.py`**: Main training script for YOLO-WS that initializes the `CustomComputeLoss` class (line 370) and implements the training pipeline
- **`src/yolov5/custom/loss.py`**: Custom loss implementation with:
  - `CustomComputeLoss` class (line 90) that uses EIoU loss and focal loss
  - `bbox_eiou` function (line 37) for improved bounding box regression
  - Focal loss implementation (lines 226-234) with gamma parameter for better handling of hard examples
- **`src/yolov5/utils/datasets.py`**: Modified to support proper handling of nested multipolygon structures in version 3, which was a critical improvement that significantly enhanced performance

### 3. Validation Implementation

- **`src/yolov5/val_wbf.py`**: Validation script with Weighted Boxes Fusion instead of traditional NMS for better handling of overlapping web elements
- **`src/yolov5/utils/metrics.py`**: Modified evaluation metrics specifically optimized for web segmentation tasks

### Key Modifications

1. **Weak Supervision Adaptations**:
   - Modified the training pipeline to support weakly supervised learning for web page segmentation
   - Implemented specialized data handling for web page screenshots and annotations
   - **Critical Improvement**: Added support for handling nested multipolygon structures in web layouts in `src/yolov5/utils/datasets.py`, which:
     - Properly recognizes and processes holes within polygons (version 3)
     - Correctly handles nested UI elements common in web interfaces
     - Significantly improved performance compared to the original paper's results by preserving the hierarchical structure of web elements

2. **Training Configuration**:
   - Custom hyperparameters optimized for web segmentation tasks
   - Modified loss functions to better handle web UI element detection through:
     - EIoU loss implementation in `src/yolov5/custom/loss.py` (line 37)
     - Focal loss with gamma parameter for better handling of class imbalance (lines 226-234)
   - Specialized data augmentation techniques for web page screenshots

3. **Model Architecture**:
   - Retained the core YOLOv5 detection framework
   - Adapted the model to better handle the structured nature of web layouts
   - Optimized for detecting UI elements with clear boundaries

## Implementation Details

### Training Implementation

The training implementation in `train_yolowsv5.py` extends the original YOLOv5 training script with the following modifications:

```python
# Custom configuration for web segmentation
from torch_config import *  # Custom torch configuration for YOLO-WS
```

The implementation follows the same training paradigm as YOLOv5 but with specialized components for web segmentation:

```
Usage - Single-GPU training:
    $ python train_yolowsv5.py --data webis-webseg-20.yaml --weights yolov5s.pt --img 640
```

### Dataset Handling

The implementation is designed to work with web segmentation datasets, particularly:

1. **Webis-Webseg-20**: The primary dataset used for training and evaluation
2. **Nano**: A smaller custom dataset for specialized testing

### Performance Considerations

Based on your research findings:

1. **Improved Handling of Nested Structures**: The implementation significantly improved performance by better handling nested multipolygon structures in web layouts
2. **Instance Segmentation vs. Object Detection**: The implementation found that instance segmentation didn't substantially outperform object detection for web segmentation tasks
3. **Training Set Size**: Performance improvements followed a logarithmic pattern with respect to training set size

## Technical Approach

The YOLO-WS implementation follows these key technical approaches:

1. **Weakly Supervised Learning**: Uses weak supervision signals to train the model effectively with limited annotations
2. **Transfer Learning**: Leverages pre-trained YOLOv5 weights as a starting point
3. **Domain-Specific Optimization**: Custom hyperparameters and training strategies optimized for web page content

## Comparison with Original Paper

Your implementation of YOLO-WS initially underperformed compared to the original paper's results, but after fixing implementation issues, particularly in handling nested multipolygon structures, you achieved better results.

## Integration with Other Components

The YOLO-WS implementation is designed to work alongside other components in your research:

1. **YOLOv5**: Standard implementation for comparison
2. **YOLOv11**: More recent YOLO version for comparison
3. **WEB-SAM**: Your adaptation of the Segment Anything Model for web segmentation

## Future Work

Based on your research plans, future work includes:

1. Additional experiments with WEBSAM
2. Exploration of webpage splitting techniques
3. Further optimization of the weak supervision approach
