# vt2-visual-webseg

Research project on web segmentation using YOLO models and WEB-SAM.

## Project Structure

```bash
.
├── data                            # Dataset directory (not tracked in git)
├── docs                            # Documentation and paper drafts
├── evaluation                      # Evaluation scripts and tools
│   ├── preprocessing_comparison    # Tools for comparing preprocessing methods
│   └── scripts                     # Evaluation scripts including inference and metrics
├── global_utils                    # Global utility functions
├── models                          # Trained models and outputs
│   ├── cormier                     # Cormier model outputs
│   ├── meier                       # Meier model outputs
│   ├── websam                      # WEB-SAM model outputs and runs
│   ├── yolov11                     # YOLOv11 model outputs and validation results
│   ├── yolov5                      # YOLOv5 model outputs and validation results
│   └── yolov5-ws                   # YOLOv5-WS model outputs
├── notebooks                       # Jupyter notebooks for analysis
├── src                             # Source code
│   ├── bcubed-f1                   # BCubed F1 score implementation
│   ├── websam                      # WEB-SAM implementation
│   │   ├── segment_anything        # SAM model implementation
│   │   ├── demo                    # Demo web application
│   │   ├── notebooks               # WEB-SAM notebooks
│   │   └── scripts                 # WEB-SAM scripts
│   └── yolov5                      # YOLOv5 implementation
│       ├── models                  # YOLOv5 model definitions
│       ├── utils                   # YOLOv5 utilities
│       └── data                    # YOLOv5 data utilities
├── .gitignore                      # Git ignore file
├── ensemble_boxes_wbf.py           # Weighted Box Fusion implementation
├── pyproject.toml                  # Project configuration
└── README.md                       # Project documentation
```

## Set up the environment for training

1. For YOLO-WS with YOLOv5 training

```bash
python3 -m venv yolov5-WS
source .yolov5-WS/bin/activate
pip install -r yolov5-requirements.txt
```

Then in the console (from the base directory of the project):

```bash
python src/yolov5/train_yolowsv5.py --data ./data/webis-webseg-20-yolo-no-tiny-segments-full/dataset.yaml --cfg src/yolov5/models/yolov5sWS.yaml --hyp src/hyp_yolows.yaml --imgs 512 --batch-size 32 --epochs 300 --project ./models/yolov5-ws --name yolov5-ws-imgsz512-no-tiny-segments-full
```

2. For regular YOLOv5 training

```bash
python3 -m venv .yolov5-WS
source .yolov5-WS/bin/activate
pip install -r yolov5-requirements.txt
```

```bash
python src/yolov5/train.py --data ./data/webis-webseg-20-yolo-no-tiny-segments-full/dataset.yaml --imgsz 512 --batch-size 32 --epochs 300 --project ./models/yolov5 --name yolov5-imgsz512-no-tiny-segments-full
```

3. For YOLO-WS with Ultralytics YOLOv11 training

```bash
python3 -m venv .ultralytics
source .ultralytics/bin/activate
pip install ultralytics
```

Then in the console (for standard parameters):

```bash
yolo detect train data=./data/webis-webseg-20-yolo-no-tiny-segments-full/dataset.yaml model=yolo11s.pt imgsz=512 batch=32 epochs=300 project=./models/yolov11 name=yolov11-imgsz512-no-tiny-segments-full
```

4. For WEB-SAM training

```bash
python3 -m venv .websam
source .websam/bin/activate
pip install -r websam-requirements.txt
```

```bash
# Full training run
python src/websam/train_websam.py --data ./data/webis-webseg-20-sam-like-on-github-10 --imgsz 1024 --batch_size 2 --epochs 20 --project ./models/websam --name test-run --max_samples 10
```

Available parameters:
- `--data`: Path to the dataset directory
- `--imgsz`: Image size for training (default: 1024)
- `--batch_size`: Batch size for training (default: 2)
- `--epochs`: Number of training epochs (default: 20)
- `--project`: Directory for saving results (default: "../../models/websam")
- `--name`: Name of the training run (default: timestamp)
- `--resume`: Path to run directory to resume training from
- `--checkpoint`: Path to SAM checkpoint (default: "../../models/websam/websam/sam_vit_b_01ec64.pth")
- `--lr`: Learning rate (default: 2e-4)
- `--freeze_encoder`: Flag to freeze encoder weights
- `--max_samples`: Maximum number of samples to use (for quick testing)
