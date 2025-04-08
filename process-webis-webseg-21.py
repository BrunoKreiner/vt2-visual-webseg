import os
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import neptune
import concurrent.futures
from tqdm import tqdm
from functools import partial
import random
import torch
import torchvision.transforms as T
import torchvision.utils as vutils

import argparse
#from src.yolowsv5 import YOLOWS, C3_WS

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import os
load_dotenv('./.env')
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import gc

# Define paths
#SOURCE_DATA = Path("./data/webis-webseg-21/webis-webseg-20-4096px")
SOURCE_DATA = Path("./data/webis-webseg-20/3988124/webis-webseg-20-screenshots")
YOLO_DATASET = Path("./data/yolo_dataset_processed").resolve()
YOLO_RUN_NAME = "./models/yolo-large/1"
CLASSES = ["webpage_segment"]

def draw_ground_truth(image_path, ground_truth_data):
    """Draw ground truth on image and save back to original location with _with_ground_truth suffix"""
    # Open the image and convert to RGBA mode first
    img = Image.open(image_path).convert('RGBA')
    draw = ImageDraw.Draw(img)
    
    segmentations = ground_truth_data['segmentations']['majority-vote']
    colors = [(255, 0, 0, 64), (0, 255, 0, 64), (0, 0, 255, 64), 
             (255, 255, 0, 64), (255, 0, 255, 64), (0, 255, 255, 64)]
    
    for i, multipolygon in enumerate(segmentations):
        color = colors[i % len(colors)]
        for polygon in multipolygon:
            for ring in polygon:
                coords = [coord for point in ring for coord in point]
                draw.polygon(coords, fill=color, outline=(0, 0, 0, 255))

    output_path = image_path.parent / f"{image_path.stem}_with_ground_truth{image_path.suffix}"
    img.save(output_path, "PNG")

def polygon_to_bbox(polygon):
    """Convert polygon to bounding box coordinates"""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def bbox_to_yolo_format(bbox, image_width, image_height):
    """Convert bounding box to YOLO format"""
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height

def process_single_website(website_folder, visualize=True):
    """Process a single website folder and return image and annotation paths"""
    screenshots = sorted(website_folder.glob("screenshot-*.png"))
    ground_truths = sorted(website_folder.glob("ground-truth-*.json"))
    
    if not screenshots or not ground_truths:
        return None, None
    
    screenshot_path = screenshots[0]
    gt_path = ground_truths[0]
    
    if visualize:
        gt_vis_path = screenshot_path.parent / f"{screenshot_path.stem}_with_ground_truth{screenshot_path.suffix}"
        if not gt_vis_path.exists():
            try:
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                draw_ground_truth(screenshot_path, gt_data)
            except Exception as e:
                tqdm.write(f"Error processing {website_folder.name}: {str(e)}")
                return None, None
    
    return screenshot_path, gt_path

def collect_dataset(source_path, limit=30, max_workers=8):
    """Collect all valid image and ground truth pairs using concurrent processing"""
    images = []
    annotations = []
    
    # Get website folders
    website_folders = [f for f in sorted(source_path.iterdir()) if f.is_dir()]
    website_folders = website_folders[:limit]
    
    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for all website folders
        future_to_folder = {
            executor.submit(process_single_website, folder): folder 
            for folder in website_folders
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=len(website_folders), desc="Collecting dataset") as pbar:
            for future in concurrent.futures.as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    img_path, ann_path = future.result()
                    if img_path is not None and ann_path is not None:
                        images.append(img_path)
                        annotations.append(ann_path)
                except Exception as e:
                    tqdm.write(f"Error processing {folder.name}: {str(e)}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "Processed": len(images), 
                    "Current": folder.name,
                    "Success Rate": f"{len(images)}/{pbar.n}"
                })
    
    return images, annotations

def process_single_sample(args):
    """Process a single image-annotation pair for YOLO dataset"""
    img_path, ann_path, split_name, output_path = args
    
    try:
        # Copy image
        img_name = f"{img_path.parent.name}_{img_path.name}"
        img_dest = output_path / 'images' / split_name / img_name
        shutil.copy(img_path, img_dest)

        # Process annotations
        with open(ann_path, "r") as f:
            data = json.load(f)
            width, height = data["width"], data["height"]

            yolo_annotations = []
            for multipolygon in data["segmentations"]["majority-vote"]:
                for polygon in multipolygon:
                    outer_ring = polygon[0]
                    bbox = polygon_to_bbox(outer_ring)
                    yolo_bbox = bbox_to_yolo_format(bbox, width, height)
                    yolo_annotations.append(f"0 {' '.join(map(str, yolo_bbox))}")

        # Save YOLO format annotations
        label_name = img_name.replace('.png', '.txt')
        label_dest = output_path / 'labels' / split_name / label_name
        with open(label_dest, "w") as f:
            f.write("\n".join(yolo_annotations))
        
        return True, img_name
    
    except Exception as e:
        return False, f"Error processing {img_path.name}: {str(e)}"

def prepare_yolo_dataset(images, annotations, output_path, max_workers=8):
    """Prepare dataset in YOLO format using concurrent processing"""
    # Create directory structure
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Split dataset
    train_imgs, val_imgs, train_anns, val_anns = train_test_split(
        images, annotations, test_size=0.2, random_state=42
    )

    # Prepare arguments for processing
    process_args = []
    for split_name, (split_imgs, split_anns) in {
        'train': (train_imgs, train_anns),
        'val': (val_imgs, val_anns)
    }.items():
        process_args.extend([
            (img_path, ann_path, split_name, output_path)
            for img_path, ann_path in zip(split_imgs, split_anns)
        ])

    # Process samples concurrently with progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_sample, args) for args in process_args]
        
        success_count = 0
        with tqdm(total=len(process_args), desc="Preparing YOLO dataset") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    tqdm.write(result)  # Print error message
                
                pbar.update(1)
                pbar.set_postfix({
                    "Successful": success_count,
                    "Success Rate": f"{success_count}/{pbar.n}",
                })

    return success_count

def create_dataset_yaml(output_path):
    """Create YOLO dataset configuration file with Neptune logging"""
    yaml_content = f"""
path: {output_path}
train: images/train
val: images/val

names:
  0: webpage_segment
"""
    with open(output_path / "dataset.yaml", "w") as f:
        f.write(yaml_content.strip())

def create_neptune_callbacks():

    print(os.getenv("NEPTUNE_API_TOKEN"))

    """Create Neptune callback functions"""
    run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT_NAME"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name="YOLO-WebSeg-Training",
        tags=["yolo", "webseg", "object-detection"]
    )

    def on_pretrain_routine_start(trainer):
        """Log initial configuration and setup"""
        if hasattr(trainer, 'args'):
            for k, v in vars(trainer.args).items():
                if isinstance(v, (int, float, str, bool)):
                    run[f"parameters/{k}"] = v

    def on_train_epoch_end(trainer):
        metrics = trainer.metrics
        epoch = trainer.epoch
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                run[f"metrics/train/{k}"].append(value=v, step=epoch)

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch
        latest_model_path = trainer.last
        print(f"Epoch {epoch + 1} completed, latest model saved at {latest_model_path}")

        if (epoch + 1) % 2000 == 0:
            # Load the latest saved model
            model = YOLO(latest_model_path)
            
            # Get validation images
            val_images_path = YOLO_DATASET / 'images' / 'val'
            all_images = [os.path.join(val_images_path, img) for img in os.listdir(val_images_path)
                        if img.endswith(('.jpg', '.png', '.jpeg'))]
            random.seed(42)
            random_images = random.sample(all_images, 9)
            
            # Perform inference
            results = model(random_images)
            
            # Create epoch folder
            img_save_path = YOLO_RUN_NAME + f"/epoch_{epoch + 1}_val_detection.png"
            
            # Adjust figure size for larger images
            fig, axs = plt.subplots(3, 3, figsize=(20, 20))  # Larger figsize for bigger images
            
            for idx, result in enumerate(results):
                # Display results with bounding boxes
                result_image = result.plot()  # Render the image with bounding boxes
                ax = axs[idx // 3, idx % 3]
                ax.imshow(result_image)
                ax.axis('off')
                
                # Optionally adjust spacing between images
                ax.set_position([0.05 + 0.33 * (idx % 3), 0.66 - 0.33 * (idx // 3), 0.3, 0.3])

            # Save the figure with less compression
            plt.savefig(img_save_path, dpi=600, bbox_inches='tight')
            plt.show()

        # At the end of your function
        torch.cuda.empty_cache()



    def on_val_end(validator):
        """Log validation metrics"""
        # Access metrics through validator.metrics
        metrics = validator.metrics
        if hasattr(validator, 'epoch'):
            epoch = validator.epoch
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    run[f"metrics/val/{k}"].append(value=v, step=epoch)

    def on_train_end(trainer):
        """Log final model performance and close Neptune run"""
        # Log best metrics
        epoch = trainer.epoch
        if hasattr(trainer, 'best_fitness'):
            run["metrics/best_fitness"] = trainer.best_fitness
        
        # Log final model path if available
        if hasattr(trainer, 'best_model_path'):
            run["artifacts/best_model_path"].upload(trainer.best_model_path)

        # Close Neptune run
        run.stop()

    return {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_val_end": on_val_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end
    }

def check_dataset_size(dataset_path, required_size):
    """Check if dataset has enough images"""
    train_path = dataset_path / 'images' / 'train'
    if not train_path.exists():
        return False
    val_path = dataset_path / 'images' / 'val'
    if not val_path.exists():
        return False
    
    print(len(list(val_path.glob('*.png'))))
    
    current_size = len(list(train_path.glob('*.png'))) + len(list(val_path.glob('*.png')))
    return current_size >= required_size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['yolo', 'yolo-ws'], 
                      default='yolo', help='Choose model type: yolo or yolo-ws')
    parser.add_argument('--transfer_learning', action='store_true',
                      help='Use transfer learning from DocLayNet')
    return parser.parse_args()

def main():
    args = parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()
    required_images = 500
    
    if not check_dataset_size(YOLO_DATASET, required_images):
        print(f"Dataset doesn't have enough images (need {required_images}), preparing dataset...")
        
        if YOLO_DATASET.exists():
            shutil.rmtree(YOLO_DATASET)
        
        print("Collecting dataset...")
        images, annotations = collect_dataset(
            SOURCE_DATA, 
            limit=required_images,
            max_workers=16
        )
        
        # Prepare YOLO dataset
        print("Preparing YOLO dataset...")
        prepare_yolo_dataset(images, annotations, YOLO_DATASET)
        
        # Create dataset configuration
        create_dataset_yaml(YOLO_DATASET)
    else:
        print(f"Dataset already exists with sufficient images...")
    
    # Initialize model based on argument
    if args.model_type == 'yolo-ws':
        print("Starting YOLO-WS training...")
        model = YOLO("./models/yolov5n.pt", load_weights=True)
        print(f"Model type: {type(model)}")
        print(f"Model detection class: {type(model.model)}")
        print(f"Model detection model class: {type(model.model.model)}")
        print(f"Trainer class: {type(model.trainer)}")
        print(f"Predictor class: {type(model.predictor)}")
        print(f"Loss function: {model.loss.__name__}")
        #print(f"Validator class: {type(model.validator)}")
        
        """if args.transfer_learning:
            print("Starting transfer learning from DocLayNet...")
            # First train on DocLayNet
            results = model.train(
                data=str(DOCLAYNET_YAML),
                epochs=50,
                imgsz=640,
                batch=32,
                workers=16,
                project=YOLO_RUN_NAME,
                name="pretrain"
            )
            
            # Freeze backbone layers for transfer learning
            print("Freezing backbone layers...")
            for param in model.model.backbone.parameters():
                param.requires_grad = False"""
    else:
        print("Starting YOLO training...")
        model = YOLO("yolo11n.pt")

    def batch_progress_callback(trainer):
        if trainer.epoch < 1:  # Only during first epoch
            try:
                print(f"\nProcessing batch")
                if hasattr(trainer, 'loss'):
                    print(f"Loss: {trainer.loss}")
                    
                # Check skip connections
                model = trainer.model.model if hasattr(trainer.model, 'model') else trainer.model
                for i, m in enumerate(model):
                    if hasattr(m, 'panet_stage'):
                        print(f"Layer {i} ({m.panet_stage})")
            except Exception as e:
                print(f"Error in callback: {e}")

    def epoch_progress_callback(trainer):
        print(f"\n=== Epoch {trainer.epoch} completed ===")
        print(f"Box loss: {trainer.loss.box:.3f}")
        print(f"Classification loss: {trainer.loss.cls:.3f}")
        print(f"DFL loss: {trainer.loss.dfl:.3f}")
    
    # Get Neptune callbacks
    callbacks = create_neptune_callbacks()

    # Add our debugging callbacks to the existing dictionary
    #callbacks['on_train_batch_end'] = on_train_batch_end
    #callbacks['on_batch_end'] = batch_progress_callback
    #callbacks['on_train_start'] = training_start_callback
    callbacks['on_epoch_end'] = epoch_progress_callback

    # Now the callbacks dictionary contains both Neptune and debugging callbacks
    for event, callback in callbacks.items():
        model.add_callback(event, callback)

    def forward_hook(module, input, output):
        print(f"Forward pass through {type(module).__name__}")
        print(f"Input shape: {input[0].shape}")
        print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")

    # Register the hook on the model itself instead of a specific layer
    model.register_forward_hook(forward_hook)

    # Train model
    results = model.train(
        data=str(YOLO_DATASET / "dataset.yaml"),
        epochs=25,
        imgsz=640,
        batch=32,        
        workers=16,
        project=YOLO_RUN_NAME,
        name="train",
        verbose = True
    )

if __name__ == "__main__":
    main()