# WEB-SAM: Modifications to the Segment Anything Model

This document details the modifications made to the original Segment Anything Model (SAM) to create WEB-SAM for web page segmentation tasks.

## Base Repository Information

- **Original Repository**: Facebook Research's Segment Anything Model (SAM)
- **URL**: https://github.com/facebookresearch/segment-anything.git
- **Commit ID**: dca509fe793f601edb92606367a655c15ac00fdf
- **Commit Date**: September 18, 2024
- **Branch**: main

The implementation in `/src/websam/` is based on this specific commit of the original repository. For reference, the original implementation can be accessed directly via:
```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
git checkout dca509fe793f601edb92606367a655c15ac00fdf
```

## Key Architectural Modifications

### 1. Training Capability Enhancement

The original SAM model was primarily designed for inference. To enable training for web segmentation tasks, the following modification was made:

```diff
- @torch.no_grad()
+ #@torch.no_grad() comment this out for training
  def forward(
      self,
      batched_input: List[Dict[str, Any]],
```

This critical change allows for gradient computation during training, enabling fine-tuning of the model on web page data.

### 2. Model Building Flexibility

Modified the model building functions in `build_sam.py` to add parameters for training control:

```diff
- def build_sam_vit_h(checkpoint=None):
+ def build_sam_vit_h(checkpoint=None, strict_weights = True, freeze_encoder= True):
      return _build_sam(
          encoder_embed_dim=1280,
          encoder_depth=32,
          encoder_num_heads=16,
          encoder_global_attn_indexes=[7, 15, 23, 31],
          checkpoint=checkpoint,
+         strict_weights = strict_weights,
+         freeze_encoder= freeze_encoder
      )
```

Similar changes were applied to all model variants (vit_h, vit_l, vit_b). These modifications provide:
- `strict_weights` parameter: Allows loading partial weights, useful for transfer learning
- `freeze_encoder` parameter: Controls whether the encoder is trainable, enabling efficient fine-tuning

### 3. Web-Specific Components

Added specialized components for web page segmentation in `src/websam/segment_anything/modeling/image_encoder.py`:

1. **Edge Detection and Processing**: 
   - `EdgeComponentsTune` class (lines 440-518) implements Sobel operators for enhanced edge detection in web layouts
   - Uses fixed Sobel kernels (lines 446-470) specifically calibrated for web element boundaries
   - Includes patch partitioning and dimension reduction for efficient processing

2. **Patch Embedding Tuning**: 
   - `PatchEmbeddingTune` class (lines 432-438) provides specialized tuning for web content patches
   - Implements linear projection with scale factor 8 for dimension reduction

3. **Adapter Modules**: 
   - `Adapter` class (lines 520-554) enables efficient fine-tuning without modifying all parameters
   - Integrates edge features with image encoder features
   - Uses bottleneck architecture with MLPktune and MLPup for parameter efficiency

### 4. Image Encoder Enhancements for Web Content

Added specialized components to the image encoder for better handling of web page layouts:

```diff
+ #WEB-SAM: Add PatchEmbeddingTune and EdgeComponentsTune
+ self.patch_embedding_tune = PatchEmbeddingTune(embed_dim = embed_dim, scale_factor=8)
+ self.edge_component_tune = EdgeComponentsTune(
+     kernel_size=(16, 16),
+     stride=(16, 16),
+     padding=(0, 0),
+     in_chans=1,
+     embed_dim=embed_dim,
+     scale_factor=8  # As recommended by the paper, mu=8 → output dim becomes 768/8 = 96.
+ )
```

Also implemented adapter modules for efficient fine-tuning:

```diff
+ if adapter_dim is None:
+     adapter_dim = embed_dim // 2
+
+ self.adapters = nn.ModuleList([
+     Adapter(
+         embed_dim=embed_dim,
+         adapter_dim=adapter_dim,  # From paper's implementation details
+         scale_factor=8
+     ) for _ in range(depth)
+ ])
```

## Implementation Details

### Modified Files

The following key files were modified or created to implement WEB-SAM:

#### 1. Core Model Files
- **`src/websam/segment_anything/modeling/sam.py`**: Modified to enable training by removing `@torch.no_grad()` decorator
- **`src/websam/segment_anything/build_sam.py`**: Added parameters for training control (strict_weights, freeze_encoder)
- **`src/websam/segment_anything/modeling/image_encoder.py`**: Enhanced with websam components

#### 2. Training Scripts
- **`src/websam/train_websam.py`**: Python script for training WEB-SAM with custom loss functions
- **`src/websam/train-websam.ipynb`**: Jupyter notebook version for interactive training and visualization

#### 3. Evaluation Implementation
- **`src/bcubed_f1_websam.py`**: Implementation of the B-Cubed F1 metric for evaluating segmentation quality

### Added Components

1. **PatchEmbeddingTune**: Specialized tuning for patch embeddings with scale factor 8
2. **EdgeComponentsTune**: 
   - Kernel size: 16×16
   - Designed to better detect edges and boundaries in web layouts
   - Uses scale factor 8 for dimension reduction
3. **Adapter Modules**:
   - Added to each transformer block
   - Default adapter dimension: embed_dim/2
   - Scale factor: 8
   - Enables efficient fine-tuning without modifying all parameters

### Training Implementation

The training implementation includes:

- **`src/websam/train_websam.py`**: Main training script with:
  - `BCEIoULoss` class (lines 141-160) that combines Binary Cross Entropy and IoU loss for improved segmentation quality:
    ```python
    # Binary Cross Entropy component for pixel-level classification
    bce_loss = self.bce(inputs, targets)
    
    # IoU component for region-level segmentation quality
    inputs_sigmoid = torch.sigmoid(inputs)
    intersection = (inputs_sigmoid * targets).sum((1, 2))
    total = (inputs_sigmoid + targets).sum((1, 2))
    union = total - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    iou_loss = 1 - iou.mean()
    
    # Combined loss balances pixel-level and region-level objectives
    total_loss = bce_loss + iou_loss.mean()
    ```
  - `calculate_metrics` function (lines 202-237) for computing PB3, RB3, F*B3, F1, and Pixel Accuracy
  - Training loop with mixed precision support using `torch.cuda.amp.GradScaler` (line 23)
  - Visualization functions for monitoring training progress

- **Support for multipolygon handling**: Implemented proper hole recognition (version 3) in the data loading pipeline, which significantly improved performance compared to the original implementation

- **`src/bcubed_f1_websam.py`**: Implementation of the B-Cubed F1 metric for evaluating segmentation quality with proper handling of overlapping segments

- **Configurable training parameters**:
  - Learning rate scheduling with `CosineAnnealingLR` (line 17)
  - Gradient accumulation for effective batch size control (line 22)
  - Checkpoint saving and resuming capabilities

## Technical Approach

The modifications follow a parameter-efficient fine-tuning approach:
1. The original SAM architecture is largely preserved
2. Adapter modules are inserted to enable efficient adaptation to web content
3. Edge component tuning is added to better handle the structured nature of web layouts
4. Training is enabled by removing the no_grad constraint

This approach allows WEB-SAM to leverage the powerful foundation of SAM while specializing for the unique challenges of web page segmentation, including:
- Structured layouts with clear boundaries
- Hierarchical organization of content
- Text-heavy regions that require different handling than natural images

## Performance Considerations

The scale factor of 8 was chosen based on empirical testing, providing a good balance between:
- Model capacity (enough parameters to learn web-specific features)
- Training efficiency (reduced parameter count compared to full fine-tuning)
- Memory usage (important for processing high-resolution web page screenshots)

The edge component tuning specifically addresses the importance of detecting UI element boundaries in web pages, which are often more structured and regular than boundaries in natural images.
