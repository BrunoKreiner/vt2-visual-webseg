import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
SOURCE_DATA = Path("../../data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20")
GROUND_TRUTH_DATA = Path("../../data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20")
OUTPUT_DIR = Path("../preprocessing_comparison")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum area threshold for the improved method
MIN_RELATIVE_AREA = 0.0001  # 0.01% of image area

def polygon_to_bbox(polygon):
    """Original method: Convert polygon to bounding box coordinates"""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def bbox_to_yolo_format(bbox, image_width, image_height):
    """Original method: Convert bounding box to YOLO format"""
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height

def polygon_to_yolo_improved(multipolygon, img_width, img_height):
    """Improved method: Convert multipolygon to YOLO format"""
    try:
        all_x = []
        all_y = []
        
        for polygon in multipolygon:
            for ring in polygon:
                for point in ring:
                    x, y = point
                    all_x.append(x)
                    all_y.append(y)

        if not all_x or not all_y:
            return None
            
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Ensure non-zero width and height
        width = max(x_max - x_min, 1)
        height = max(y_max - y_min, 1)    
        
        # Convert to YOLO format
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = width / img_width
        height = height / img_height
        relative_area = (width * height)  # Already in normalized coordinates
        
        if relative_area < MIN_RELATIVE_AREA:
            return None
        
        # Validate values are between 0 and 1
        if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
            return None
            
        return [x_center, y_center, width, height]
        
    except Exception as e:
        logging.error(f"Error converting polygon to YOLO format: {str(e)}")
        return None

def process_sample_original(website_id):
    """Process a sample using the original method"""
    img_path = SOURCE_DATA / website_id / "screenshot.png"
    gt_path = GROUND_TRUTH_DATA / website_id / "ground-truth.json"
    
    if not img_path.exists() or not gt_path.exists():
        return None
    
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        with open(gt_path, "r") as f:
            data = json.load(f)
        
        yolo_annotations = []
        for multipolygon in data["segmentations"]["majority-vote"]:
            for polygon in multipolygon:
                outer_ring = polygon[0]
                bbox = polygon_to_bbox(outer_ring)
                yolo_bbox = bbox_to_yolo_format(bbox, img_width, img_height)
                yolo_annotations.append(yolo_bbox)
        
        return {
            'id': website_id,
            'img_path': img_path,
            'annotations': yolo_annotations,
            'width': img_width,
            'height': img_height
        }
    
    except Exception as e:
        logging.error(f"Error processing sample {website_id} with original method: {str(e)}")
        return None

def process_sample_improved(website_id):
    """Process a sample using the improved method"""
    img_path = SOURCE_DATA / website_id / "screenshot.png"
    gt_path = GROUND_TRUTH_DATA / website_id / "ground-truth.json"
    
    if not img_path.exists() or not gt_path.exists():
        return None
    
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        with open(gt_path, "r") as f:
            data = json.load(f)
        
        yolo_annotations = []
        for segment in data["segmentations"]["majority-vote"]:
            bbox = polygon_to_yolo_improved(segment, img_width, img_height)
            if bbox is not None:
                yolo_annotations.append(bbox)
        
        return {
            'id': website_id,
            'img_path': img_path,
            'annotations': yolo_annotations,
            'width': img_width,
            'height': img_height
        }
    
    except Exception as e:
        logging.error(f"Error processing sample {website_id} with improved method: {str(e)}")
        return None

def visualize_comparison(website_id):
    """Visualize the comparison between original and improved methods"""
    original_result = process_sample_original(website_id)
    improved_result = process_sample_improved(website_id)
    
    if original_result is None or improved_result is None:
        logging.error(f"Failed to process sample {website_id}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Load image
    img = cv2.imread(str(original_result['img_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Original method visualization
    ax1.imshow(img)
    ax1.set_title(f"Original Method: {len(original_result['annotations'])} boxes")
    
    for bbox in original_result['annotations']:
        # Convert YOLO format to pixel coordinates
        x_center, y_center, w, h = bbox
        x_center *= original_result['width']
        y_center *= original_result['height']
        w *= original_result['width']
        h *= original_result['height']
        
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        w = int(w)
        h = int(h)
        
        # Create rectangle patch
        rect = plt.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    
    # Improved method visualization
    ax2.imshow(img)
    ax2.set_title(f"Improved Method: {len(improved_result['annotations'])} boxes")
    
    for bbox in improved_result['annotations']:
        # Convert YOLO format to pixel coordinates
        x_center, y_center, w, h = bbox
        x_center *= improved_result['width']
        y_center *= improved_result['height']
        w *= improved_result['width']
        h *= improved_result['height']
        
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        w = int(w)
        h = int(h)
        
        # Create rectangle patch
        rect = plt.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
    
    # Save figure
    output_path = OUTPUT_DIR / f"{website_id}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved comparison visualization for {website_id} to {output_path}")
    return output_path

def main():
    # Get a list of website IDs
    website_folders = [f.name for f in SOURCE_DATA.iterdir() if f.is_dir()]
    
    # Randomly select 5 websites to visualize
    if len(website_folders) > 5:
        selected_websites = random.sample(website_folders, 20)
        
    else:
        selected_websites = website_folders
    
    visualized_paths = []
    for website_id in selected_websites:
        output_path = visualize_comparison(website_id)
        if output_path:
            visualized_paths.append(output_path)
    
    logging.info(f"Generated {len(visualized_paths)} comparison visualizations")
    for path in visualized_paths:
        logging.info(f"- {path}")

if __name__ == "__main__":
    main()
