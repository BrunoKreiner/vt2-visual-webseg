import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from websam.segment_anything.build_sam import build_sam_vit_b
from shapely.geometry import box, Polygon, mapping
import shutil
import argparse

def load_websam_model(checkpoint_path, sam_base_path, device='cuda'):
    """Load and prepare the WebSAM model"""
    # First load the base SAM model
    sam = build_sam_vit_b(
        checkpoint=sam_base_path,
        strict_weights=False,
        freeze_encoder=True
    )
    
    # Then load the WebSAM weights
    checkpoint = torch.load(checkpoint_path)
    sam.load_state_dict(checkpoint['model_state_dict'])
    sam.to(device)
    sam.eval()
    
    return sam

def generate_predictions(model, data_dir, output_dir, device='cuda'):
    """Generate predictions using WebSAM model and save them in the B-Cubed format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the validation image paths
    val_img_dir = os.path.join(data_dir, 'val', 'images')
    val_box_dir = os.path.join(data_dir, 'val', 'boxes')
    val_mask_dir = os.path.join(data_dir, 'val', 'masks')
    
    # Process each image
    for img_file in tqdm(os.listdir(val_img_dir), desc="Generating WebSAM predictions"):
        if not img_file.endswith(('.jpg', '.png')):
            continue
        
        # Load image and boxes
        img_path = os.path.join(val_img_dir, img_file)
        box_path = os.path.join(val_box_dir, img_file.replace('.png', '.npy'))
        mask_path = os.path.join(val_mask_dir, img_file.replace('.png', '.npy'))
        
        # Skip if any of the files are missing
        if not (os.path.exists(img_path) and os.path.exists(box_path) and os.path.exists(mask_path)):
            print(f"Skipping {img_file} due to missing files")
            continue
        
        # Load and preprocess
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load bounding boxes
        boxes = np.load(box_path)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).to(device)
        
        # Load ground truth masks for reference
        gt_masks = np.load(mask_path)
        
        # Get page ID from filename
        page_id = os.path.splitext(img_file)[0]
        
        # Create model input
        image_tensor = torch.tensor(image).permute(2, 0, 1).float().to(device)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
            
        batched_input = [{
            "image": image_tensor,
            "boxes": boxes_tensor,
            "original_size": (height, width)
        }]
        
        # Generate predictions
        with torch.no_grad():
            outputs = model(batched_input, multimask_output=False)
        
        # Convert predictions to B-Cubed format
        predictions = []
        for i, mask_output in enumerate(outputs[0]["low_res_logits"]):
            # Process each mask prediction
            mask = mask_output.sigmoid().cpu().numpy() > 0.5
            
            # Resize mask to original image size
            mask_full_res = cv2.resize(
                mask.astype(np.uint8), 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convert binary mask to polygon
            contours, _ = cv2.findContours(
                mask_full_res.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Take the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 100:  # Skip very small predictions
                    # Convert contour to polygon
                    polygon_points = largest_contour.reshape(-1, 2).tolist()
                    if len(polygon_points) > 3:  # Need at least 3 points for a valid polygon
                        try:
                            poly = Polygon(polygon_points)
                            if poly.is_valid:
                                predictions.append({
                                    "polygon": mapping(poly),
                                    "tagType": f"segment_{i}"  # Tag with segment ID
                                })
                        except Exception as e:
                            print(f"Error creating polygon for {page_id}, segment {i}: {e}")
        
        # Process ground truth masks (for comparison)
        ground_truth = []
        for i, gt_mask in enumerate(gt_masks):
            # Skip empty masks
            if gt_mask.max() == 0:
                continue
                
            # Convert binary mask to polygon
            contours, _ = cv2.findContours(
                gt_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Take the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 100:  # Skip very small ground truths
                    # Convert contour to polygon
                    polygon_points = largest_contour.reshape(-1, 2).tolist()
                    if len(polygon_points) > 3:  # Need at least 3 points for a valid polygon
                        try:
                            poly = Polygon(polygon_points)
                            if poly.is_valid:
                                ground_truth.append({
                                    "polygon": mapping(poly),
                                    "tagType": f"segment_{i}"  # Tag with segment ID
                                })
                        except Exception as e:
                            print(f"Error creating ground truth polygon for {page_id}, segment {i}: {e}")
        
        # Create JSON structure with annotator IDs as keys
        result_json = {
            "id": page_id,
            "height": height,
            "width": width,
            "segmentations": {
                "annotator1": predictions,    # WebSAM predictions
                "annotator2": ground_truth    # Ground truth
            }
        }
        
        # Save to JSON file
        with open(os.path.join(output_dir, f'{page_id}.json'), 'w') as f:
            json.dump(result_json, f, indent=2)

def prepare_folder_structure(predictions_dir, original_images_dir, output_base_dir, annotation_postfix='FC'):
    """Prepare the folder structure expected by B-Cubed F1 code"""
    # Loop through all prediction files
    for json_file in tqdm(os.listdir(predictions_dir), desc="Preparing folder structure"):
        if not json_file.endswith('.json'):
            continue
            
        page_id = os.path.splitext(json_file)[0]
        
        # Create folder for each page
        page_dir = os.path.join(output_base_dir, page_id)
        os.makedirs(page_dir, exist_ok=True)
        
        # Copy the annotation file with proper naming
        src_json = os.path.join(predictions_dir, json_file)
        dst_json = os.path.join(page_dir, f'annotations_{annotation_postfix}.json')
        
        # Read the JSON to ensure it has the right format
        with open(src_json, 'r') as f:
            json_data = json.load(f)
        
        # Write to the destination
        with open(dst_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Copy the corresponding image
        src_img = os.path.join(original_images_dir, f"{page_id}.png")
        if not os.path.exists(src_img):
            # Try with jpg extension
            src_img = os.path.join(original_images_dir, f"{page_id}.jpg")
        
        if os.path.exists(src_img):
            dst_img = os.path.join(page_dir, "screenshot.png")
            # If source is jpg, convert to png
            if src_img.endswith('.jpg'):
                img = cv2.imread(src_img)
                cv2.imwrite(dst_img, img)
            else:
                shutil.copy(src_img, dst_img)
        else:
            print(f"Warning: No image found for {page_id}")

def main():
    parser = argparse.ArgumentParser(description='Generate WebSAM predictions and prepare for B-Cubed evaluation')
    parser.add_argument('--websam_checkpoint', type=str, default='../models/websam/websam/run_20250225_163753/models/checkpoint_epoch_20.pth',
                       help='Path to the WebSAM checkpoint')
    parser.add_argument('--sam_base', type=str, default='../models/websam/websam/sam_vit_b_01ec64.pth',
                       help='Path to the base SAM checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data/webis-webseg-20-sam-full',
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='../data/websam_predictions_for_webisseg',
                       help='Path to save the predictions')
    parser.add_argument('--metrics_dir', type=str, default='../data/websam_for_webisseg_metrics',
                       help='Path to prepare the B-Cubed metrics structure')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading WebSAM model...")
    model = load_websam_model(args.websam_checkpoint, args.sam_base, device)
    
    # Generate predictions
    print("Generating predictions...")
    os.makedirs(args.output_dir, exist_ok=True)
    generate_predictions(model, args.data_dir, args.output_dir, device)
    
    # Prepare folder structure for B-Cubed evaluation
    print("Preparing folder structure for B-Cubed evaluation...")
    os.makedirs(args.metrics_dir, exist_ok=True)
    prepare_folder_structure(
        args.output_dir,
        os.path.join(args.data_dir, 'val', 'images'),
        args.metrics_dir
    )
    
    print(f"Done! Now you can run B-Cubed evaluation on {args.metrics_dir}")
    print("Example command:")
    print(f"cd ./src/bcubed-f1/ && python main.py --folder_path ../../{args.metrics_dir} --file_postfix 'FC' --operation 'prediction' --pixel_based")

if __name__ == "__main__":
    main()