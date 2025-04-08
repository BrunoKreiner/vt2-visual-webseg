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
SOURCE_DATA = Path("./data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20")
GROUND_TRUTH_DATA = Path("./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20")
YOLO_DATASET = Path("./data/webis-webseg-20-yolo-small").resolve()
YOLO_RUN_NAME = "./models/yolo-large/1"
CLASSES = ["webpage_segment"]
LIMIT = None

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
    # Direct path to screenshot
    screenshot_path = website_folder / "screenshot.png"
    
    if not screenshot_path.exists():
        print(f"Screenshot not found at: {screenshot_path}")
        return None, None
    
    website_id = website_folder.name
    gt_base = Path("./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20")
    gt_path = gt_base / website_id / f"ground-truth.json"
    
    if not gt_path.exists():
        print(f"Ground truth not found at: {gt_path}")
        return None, None
    
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

def collect_dataset(source_path, ground_truth_path, limit=30, max_workers=8):
    """Collect all valid image and ground truth pairs using concurrent processing"""
    images = []
    annotations = []
    
    
    # Get website folders
    website_folders = [f for f in sorted(source_path.iterdir()) if f.is_dir()]

    print(website_folders)

    if limit is not None:
        website_folders = website_folders[:limit]
    
    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for all website folders
        future_to_folder = {
            executor.submit(process_single_website, folder, ground_truth_path): folder 
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

def main():
    
    if YOLO_DATASET.exists():
        shutil.rmtree(YOLO_DATASET)
    
    print("Collecting dataset...")
    images, annotations = collect_dataset(
        SOURCE_DATA, 
        GROUND_TRUTH_DATA,
        limit=500,
        max_workers=16
    )
    
    # Prepare YOLO dataset
    print("Preparing YOLO dataset...")
    prepare_yolo_dataset(images, annotations, YOLO_DATASET)
    
    # Create dataset configuration
    create_dataset_yaml(YOLO_DATASET)

if __name__ == "__main__":
    main()