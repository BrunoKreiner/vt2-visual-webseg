import os
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import concurrent.futures
from tqdm import tqdm
import cv2

def polygon_to_mask(multipolygon, image_size):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for polygon in multipolygon:
        for ring in polygon:
            # Flatten coordinates while preserving pairs
            coords = []
            for point in ring:
                coords.extend(point)
            draw.polygon(coords, fill=1)
            
    return np.array(mask)

def polygon_to_bbox(polygon):
   """Convert polygon to XYXY bbox"""
   x_coords = [point[0] for point in polygon]
   y_coords = [point[1] for point in polygon]
   return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

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

def collect_single_website(website_folder, visualize=True):
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

    if limit is not None:
        website_folders = website_folders[:limit]
    
    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create futures for all website folders
        future_to_folder = {
            executor.submit(collect_single_website, folder, ground_truth_path): folder 
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

def resize_image_and_labels(image, masks, boxes, target_size=(1024, 1024)):
    """Resize image, masks, and bounding boxes to the target size."""
    orig_size = image.size  # (width, height)
    image = image.resize(target_size, Image.LANCZOS)
    
    scale_x = target_size[0] / orig_size[0]
    scale_y = target_size[1] / orig_size[1]
    
    resized_masks = [cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    resized_boxes = [[int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)] for x, y, w, h in boxes]
    
    return image, resized_masks, resized_boxes

def resize_image_and_labels(image, masks, boxes, target_size=(1024, 1024)):
    """Resize image, masks, and bounding boxes to the target size."""    
    orig_size = image.size  # (width, height)
    image = image.resize(target_size, Image.LANCZOS)
    
    scale_x = target_size[0] / orig_size[0]
    scale_y = target_size[1] / orig_size[1]
    
    # Convert each mask to np.uint8 before resizing
    resized_masks = [cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST) 
                     for mask in masks]
    
    # Convert bounding box from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height) then scale
    resized_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        new_xmin = int(xmin * scale_x)
        new_ymin = int(ymin * scale_y)
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        resized_boxes.append([new_xmin, new_ymin, new_width, new_height])
    
    return image, resized_masks, resized_boxes
def visualize_sample(image, masks, boxes, save_path):
    """Save an image with masks and bounding boxes overlayed."""
    # Convert image to RGB to avoid channel mismatch issues (if image is in RGBA)
    overlay = np.array(image.convert("RGB"))
    
    for mask in masks:
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        # This will correctly overlay where mask > 0 in the resized 1024x1024 mask
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + color * 0.5).astype(np.uint8)
    
    image_with_bboxes = overlay.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imwrite(str(save_path), cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR))

def process_single_sample(img_path, ann_path, split_name, output_path):
    """Process single sample for SAM training with resizing."""
    try:
        img = Image.open(img_path)
        with open(ann_path, "r") as f:
            data = json.load(f)
            
        masks = []
        boxes = []
        for multipolygon in data["segmentations"]["majority-vote"]:
            mask = polygon_to_mask(multipolygon, (data["width"], data["height"]))
            bbox = polygon_to_bbox(multipolygon[0][0])
            masks.append(mask)
            boxes.append(bbox)
        
        img, masks, boxes = resize_image_and_labels(img, masks, boxes)
        
        os.makedirs(output_path / split_name / "masks", exist_ok=True)
        os.makedirs(output_path / split_name / "boxes", exist_ok=True)
        os.makedirs(output_path / split_name / "images", exist_ok=True)
        
        unique_id = img_path.parent.name
        np.save(output_path / split_name / "masks" / f"{unique_id}.npy", masks)
        np.save(output_path / split_name / "boxes" / f"{unique_id}.npy", boxes)
        img.save(output_path / split_name / "images" / f"{unique_id}.png")
        
        # check if train and if directory exists
        if split_name == "val" and not (output_path / "sample_visualization.png").exists():
            visualize_sample(img, masks, boxes, output_path / "sample_visualization.png")
        
        return True, img_path.name
        
    except Exception as e:
        return False, str(e)

def prepare_sam_dataset(images, annotations, output_path, max_workers=8):
   """Prepare dataset in SAM format using concurrent processing"""
   # Split dataset
   train_imgs, val_imgs, train_anns, val_anns = train_test_split(
       images, annotations, test_size=0.2, random_state=42
   )

   process_args = []
   for split_name, (split_imgs, split_anns) in {
       'train': (train_imgs, train_anns),
       'val': (val_imgs, val_anns)
   }.items():
       process_args.extend([
           (img_path, ann_path, split_name, output_path)
           for img_path, ann_path in zip(split_imgs, split_anns)
       ])
   print(f"Processing {len(process_args)} samples")
   with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
       futures = [executor.submit(process_single_sample, *args) for args in process_args]
       
       success_count = 0
       with tqdm(total=len(process_args), desc="Preparing SAM dataset") as pbar:
           for future in concurrent.futures.as_completed(futures):
               success, result = future.result()
               if success:
                   success_count += 1
               else:
                   tqdm.write(result)
               
               pbar.update(1)
               pbar.set_postfix({
                   "Successful": success_count,
                   "Success Rate": f"{success_count}/{pbar.n}"
               })

   return success_count

def main():
   SOURCE_DATA_PATH = Path("./data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20")
   GROUND_TRUTH_DATA_PATH = Path("./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20")
   SAM_DATASET_OUTPUT_PATH = Path("./data/webis-webseg-20-sam-full").resolve()
   
   if SAM_DATASET_OUTPUT_PATH.exists():
       shutil.rmtree(SAM_DATASET_OUTPUT_PATH)
   
   images, annotations = collect_dataset(SOURCE_DATA_PATH, GROUND_TRUTH_DATA_PATH, limit=None)
   prepare_sam_dataset(images, annotations, SAM_DATASET_OUTPUT_PATH)

if __name__ == "__main__":
   main()