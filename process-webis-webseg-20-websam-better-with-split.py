import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import cv2
import gc
from src.image_split_algorithm import split_tall_image

MIN_RELATIVE_AREA = 0.0001  # 0.01% of image area

class WebisToWebSAMConverter:
    def __init__(self,
                 source_path,
                 ground_truth_path,
                 output_dir='./data/webis-webseg-20-sam-full',
                 num_images=None,
                 full_validation_set=False,
                 val_split=0.2,
                 image_size=1024):
        
        self.source_path = Path(source_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.val_split = val_split
        self.full_validation_set = full_validation_set
        self.image_size = image_size
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{self.output_dir}/conversion.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_directories(self):
        """Create necessary directories for WebSAM dataset"""
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'boxes').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    def validate_image(self, img_path):
        """Validate image file"""
        try:
            img = Image.open(img_path)
            # Check for minimum dimensions
            if img.size[0] < 10 or img.size[1] < 10:
                print(f"Image too small: {img_path}")
                return False, None
            return True, img
        except Exception as e:
            print(f"Error validating image {img_path}: {str(e)}")
            return False, None
            
    def validate_ground_truth(self, gt_data):
        """Validate ground truth data"""
        try:
            if 'segmentations' not in gt_data:
                return False
                
            if 'majority-vote' not in gt_data['segmentations']:
                return False
                
            if not isinstance(gt_data['segmentations']['majority-vote'], list):
                return False
                
            return True
        except Exception as e:
            print(f"Error validating ground truth: {str(e)}")
            return False

    def polygon_to_bbox(self, multipolygon, img_width, img_height, image_id=''):
        """Convert polygon to bounding box format"""
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
                print("Empty polygon detected")
                return None
                
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            # Ensure non-zero width and height
            width = max(x_max - x_min, 1)
            height = max(y_max - y_min, 1)    
            area_pixels = width * height
            
            # Convert to relative coordinates
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = width / img_width
            height = height / img_height
            relative_area = (width * height)
            
            if relative_area < MIN_RELATIVE_AREA:
                return None
            
            # Return [x_min, y_min, x_max, y_max] normalized
            x_min_norm = x_min / img_width
            y_min_norm = y_min / img_height
            x_max_norm = x_max / img_width
            y_max_norm = y_max / img_height
                
            return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
            
        except Exception as e:
            print(f"Error converting polygon to bbox format: {str(e)}")
            return None

    def create_mask_from_polygons(self, polygons, img_width, img_height):
        """Create binary mask from multipolygons with proper hole handling"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for multipolygon in polygons:
            for polygon in multipolygon:
                # First ring is always the outer boundary
                if len(polygon) > 0:
                    outer_ring = np.array(polygon[0], dtype=np.int32)
                    cv2.fillPoly(mask, [outer_ring], 1)
                    
                    # Any additional rings are holes
                    for hole_idx in range(1, len(polygon)):
                        hole = np.array(polygon[hole_idx], dtype=np.int32)
                        cv2.fillPoly(mask, [hole], 0)
        
        return mask

    def process_single_sample(self, website_id):
        """Process a single website sample"""
        try:
            img_path = self.source_path / website_id / "screenshot.png"
            gt_path = self.ground_truth_path / website_id / "ground-truth.json"
            
            if not img_path.exists() or not gt_path.exists():
                return None
                
            valid_img, img = self.validate_image(img_path)
            if not valid_img:
                return None
                
            orig_width, orig_height = img.size
            img.close()
            
            # Check if image should be split
            img_cv = cv2.imread(str(img_path))
            split_images = split_tall_image(str(img_path))
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            if not self.validate_ground_truth(gt_data):
                return None
            
            # Create the binary mask from polygons
            mask = self.create_mask_from_polygons(
                gt_data['segmentations']['majority-vote'],
                orig_width, orig_height
            )
            
            # If image was split, split the mask accordingly
            if len(split_images) > 1:
                split_height = split_images[0].shape[0]
                mask_splits = [
                    mask[:split_height, :],  # Upper half
                    mask[split_height:, :]   # Lower half
                ]
                
                # Return multiple samples
                results = []
                for i, (split_img, split_mask) in enumerate(zip(split_images, mask_splits)):
                    # Skip if split mask is empty (no segments in this part)
                    if np.sum(split_mask) < 100:  # Arbitrary threshold
                        continue
                        
                    split_id = f"{website_id}_part{i+1}"
                    
                    # Save split image for processing
                    split_img_path = self.source_path / f"{split_id}.png"
                    cv2.imwrite(str(split_img_path), split_img)
                    
                    results.append({
                        'id': split_id,
                        'img_path': split_img_path,
                        'mask': split_mask
                    })
                
                return results if results else None
            
            # If not split, return as normal
            return {
                'id': website_id,
                'img_path': img_path,
                'mask': mask
            }
            
        except Exception as e:
            print(f"Error processing sample {website_id}: {str(e)}")
            return None
        
    def save_sample(self, sample, split_name):
        """Save a single sample to disk"""
        # Resize image to target size
        img = Image.open(sample['img_path'])
        img_resized = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Resize mask to target size
        mask = sample['mask']
        mask_resized = cv2.resize(
            mask, 
            (self.image_size, self.image_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Save the image
        img_resized.save(self.output_dir / split_name / 'images' / f"{sample['id']}.png")
        
        # Save the mask as numpy array
        np.save(
            self.output_dir / split_name / 'masks' / f"{sample['id']}.npy",
            mask_resized
        )

    def convert_dataset(self):
        """Convert dataset to WebSAM format"""
        self.setup_directories()
        
        # Collect website IDs and sort them for consistent ordering
        website_ids = sorted([f.name for f in self.source_path.iterdir() if f.is_dir()])
        if self.num_images and not self.full_validation_set:
            website_ids = website_ids[:self.num_images]
        
        print(f"Found {len(website_ids)} website directories")
        
        # Split IDs into train and validation sets
        train_ids, val_ids = train_test_split(
            website_ids,
            test_size=self.val_split,
            random_state=42,
            shuffle=True
        )
        
        # Limit train IDs if num_images is specified and full_validation_set is True
        if self.num_images and self.full_validation_set:
            train_ids = train_ids[:self.num_images]
        
        # Create a dictionary mapping IDs to their split
        id_to_split = {id: 'train' for id in train_ids}
        id_to_split.update({id: 'val' for id in val_ids})
        
        # Process and save each sample
        successful_count = 0
        
        def process_and_save(website_id):
            nonlocal successful_count
            split = id_to_split[website_id]
            result = self.process_single_sample(website_id)
            
            # Handle case where result is a list of samples (image was split)
            if isinstance(result, list):
                for sample in result:
                    self.save_sample(sample, split)
                    successful_count += 1
                return len(result) > 0
            # Handle normal case (single sample)
            elif result is not None:
                self.save_sample(result, split)
                successful_count += 1
                return True
            
            return False
        
        # Process samples with parallel execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_and_save, website_id): website_id 
                    for website_id in website_ids}
            
            success_list = []
            for future in tqdm(concurrent.futures.as_completed(futures), 
                            desc="Processing and saving samples"):
                website_id = futures[future]
                success = future.result()
                success_list.append(success)
        
        train_count = len(os.listdir(self.output_dir / 'train' / 'images'))
        val_count = len(os.listdir(self.output_dir / 'val' / 'images'))
        
        # Final statistics
        print(f"""
    Dataset Summary:
    - Total processed samples: {successful_count}
    - Training samples: {train_count}
    - Validation samples: {val_count}
    """)
        
        return successful_count

def main():
    converter = WebisToWebSAMConverter(
        source_path="./data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20",
        ground_truth_path="./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20",
        output_dir="./data/webis-webseg-20-sam-big-segments-full-better-with-split",
        num_images=10,  # Process all images
        full_validation_set=False,
        val_split=0.2,
        image_size=1024
    )
    
    total_processed = converter.convert_dataset()
    print(f"Conversion completed. Total processed samples: {total_processed}")

if __name__ == "__main__":
    main()