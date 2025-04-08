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
                logging.warning(f"Image too small: {img_path}")
                return False, None
            return True, img
        except Exception as e:
            logging.error(f"Error validating image {img_path}: {str(e)}")
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
            logging.error(f"Error validating ground truth: {str(e)}")
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
                logging.warning("Empty polygon detected")
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
            logging.error(f"Error converting polygon to bbox format: {str(e)}")
            return None

    def create_mask_from_polygons(self, polygons, img_width, img_height):
        """Create binary mask from polygons"""
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for polygon in polygons:
            # Flatten the polygon points for OpenCV format
            for ring in polygon:
                points = np.array(ring, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
        
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
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            if not self.validate_ground_truth(gt_data):
                return None
            
            # Process annotations and create mask
            bboxes = []
            for segment in gt_data['segmentations']['majority-vote']:
                bbox = self.polygon_to_bbox(segment, orig_width, orig_height, website_id)
                if bbox is not None:
                    bboxes.append(bbox)
            
            if not bboxes:
                logging.warning(f"No valid bounding boxes for {website_id}")
                return None
            
            # Create the binary mask from polygons
            mask = self.create_mask_from_polygons(
                gt_data['segmentations']['majority-vote'],
                orig_width, orig_height
            )
            
            return {
                'id': website_id,
                'img_path': img_path,
                'bboxes': np.array(bboxes, dtype=np.float32),
                'mask': mask
            }
            
        except Exception as e:
            logging.error(f"Error processing sample {website_id}: {str(e)}")
            return None

    def convert_dataset(self):
        """Convert dataset to WebSAM format"""
        self.setup_directories()
        
        # Collect website IDs and sort them for consistent ordering
        website_ids = sorted([f.name for f in self.source_path.iterdir() if f.is_dir()])
        if self.num_images and not self.full_validation_set:
            website_ids = website_ids[:self.num_images]
        
        logging.info(f"Found {len(website_ids)} website directories")
        
        # Process samples with parallel execution
        processed_data = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_single_sample, website_id): website_id 
                    for website_id in website_ids}
            
            # Create a dictionary to store results in order
            results_dict = {}
            
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Processing samples"):
                website_id = futures[future]
                result = future.result()
                if result is not None:
                    results_dict[website_id] = result
            
        # Sort results by website_id to ensure consistent ordering
        processed_data = [results_dict[id] for id in sorted(results_dict.keys()) if id in results_dict]
        
        # Log statistics about processed data
        logging.info(f"Successfully processed {len(processed_data)} samples out of {len(website_ids)} total")
        
        # Split into train and validation sets with fixed ordering
        train_data, val_data = train_test_split(
            processed_data,
            test_size=self.val_split,
            random_state=42,
            shuffle=True
        )
        
        # Then limit only the training data if num_images is specified
        if self.num_images and self.full_validation_set:
            train_data = train_data[:self.num_images]

        logging.info(f"Split dataset into {len(train_data)} train and {len(val_data)} validation samples")
        
        # Save datasets
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            for sample in tqdm(split_data, desc=f"Saving {split_name} split"):
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
                
                # Save the boxes as numpy array
                np.save(
                    self.output_dir / split_name / 'boxes' / f"{sample['id']}.npy",
                    sample['bboxes']
                )
                
                # Save the mask as numpy array
                np.save(
                    self.output_dir / split_name / 'masks' / f"{sample['id']}.npy",
                    mask_resized
                )
        
        # Final statistics
        logging.info(f"""
    Dataset Summary:
    - Total processed samples: {len(processed_data)}
    - Training samples: {len(train_data)}
    - Validation samples: {len(val_data)}
    """)
        
        return len(processed_data)

def main():
    converter = WebisToWebSAMConverter(
        source_path="./data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20",
        ground_truth_path="./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20",
        output_dir="./data/webis-webseg-20-sam-big-segments-full",
        num_images=100,  # Process all images
        full_validation_set=False,
        val_split=0.2,
        image_size=1024
    )
    
    total_processed = converter.convert_dataset()
    print(f"Conversion completed. Total processed samples: {total_processed}")

if __name__ == "__main__":
    main()