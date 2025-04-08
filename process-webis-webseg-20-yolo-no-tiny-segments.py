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

MIN_RELATIVE_AREA = 0.0001  # 0.01% of image area

class WebisToYOLOConverter:
    def __init__(self,
                 source_path,
                 ground_truth_path,
                 output_dir='./data/webis-yolo-test',
                 num_images=None,
                 full_validation_set = False, # Whether to use full validation set even if images are limited (if num_images = 8, then still use 20% of 8490 webiswebseg images for validation)
                 val_split=0.2):
        
        self.source_path = Path(source_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.val_split = val_split
        self.full_validation_set = full_validation_set
        # Single class for webpage segments
        self.class_id = 0
        
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
        """Create necessary directories for YOLO dataset"""
        for split in ['train', 'val']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

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

    def analyze_box_sizes(self, processed_data):
        """Detailed analysis of box sizes in relative area percentages"""
        size_ranges = {
            'extremely_small': (0.00001, 0.0001),  # 0.001% to 0.01%
            'very_small': (0.0001, 0.001),         # 0.01% to 0.1%
            'small': (0.001, 0.01),                # 0.1% to 1%
            'medium': (0.01, 0.1),                 # 1% to 10%
            'large': (0.1, 0.5),                   # 10% to 50%
            'very_large': (0.5, float('inf'))      # > 50%
        }
        
        size_counts = {k: 0 for k in size_ranges}
        problematic_samples = {}
        
        # Analyze bounding box sizes
        total_boxes = 0
        small_boxes = 0
        box_sizes = []

        for sample in tqdm(processed_data, desc="Processing Box Sizes"):
            sample_id = sample['id']
            
            for bbox in sample['annotations']:
                # YOLO coordinates are already normalized, so just multiply width and height
                relative_area = bbox[2] * bbox[3]  # relative area as percentage of image
                box_sizes.append(relative_area)
                
                for size_name, (min_size, max_size) in size_ranges.items():
                    if min_size <= relative_area < max_size:
                        size_counts[size_name] += 1
                        if size_name == 'extremely_small':
                            if sample_id not in problematic_samples:
                                problematic_samples[sample_id] = []
                            # Convert back to pixels for logging
                            with Image.open(sample['img_path']) as img:
                                img_width, img_height = img.size
                                width_px = bbox[2] * img_width
                                height_px = bbox[3] * img_height
                                area_px = width_px * height_px
                            problematic_samples[sample_id].append((width_px, height_px, area_px))
                            self.visualize_small_boxes(sample)
                            small_boxes += 1
        
        total_boxes = sum(size_counts.values())
        logging.info("\nDetailed Box Size Distribution (relative to image area):")
        for size_name, count in size_counts.items():
            percentage = (count / (total_boxes + 1e-6)) * 100
            min_percent = size_ranges[size_name][0] * 100
            max_percent = size_ranges[size_name][1] * 100
            logging.info(f"- {size_name} ({min_percent:.3f}% to {max_percent:.3f}%): {count} boxes ({percentage:.2f}%)")
        
        print("Problematic Samples: ", len(problematic_samples))
        if problematic_samples:
            logging.warning("\nSamples with extremely small boxes (pixels):")
            for sample_id, dims in list(problematic_samples.items())[:10]:  # Show first 10
                logging.warning(f"- {sample_id}: {len(dims)} boxes with dimensions (w×h=area): "
                            f"{['%.0f×%.0f=%.0f' % (w,h,a) for w,h,a in dims]}")
        
        return total_boxes, small_boxes, box_sizes

    def polygon_to_yolo(self, multipolygon, img_width, img_height, image_id=''):
        """Convert polygon to YOLO format with debug for specific image"""
        try:
            all_x = []
            all_y = []
            
            """# Debug print only for specific image
            if image_id == '000411':
                print(f"\nProcessing image {image_id}")
                print(f"Processing multipolygon structure: {len(multipolygon)} outer elements")
                for i, polygon in enumerate(multipolygon):
                    print(f"Polygon {i} structure: {len(polygon)} elements")
                    for j, ring in enumerate(polygon):
                        print(f"Ring {j} points: {len(ring)}")
                        print(f"Points: {ring}")
                    
                    for point in ring:
                        x, y = point
                        all_x.append(x)
                        all_y.append(y)
            else:"""
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
            
            # Convert to YOLO format
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = width / img_width
            height = height / img_height
            relative_area = (width * height)  # Already in normalized coordinates
            #print("relative area: " + str(relative_area * 100) + "%")
            if relative_area < MIN_RELATIVE_AREA:
                return None
            
            # Validate values are between 0 and 1
            if not all(0 <= val <= 1 for val in [x_center, y_center, width, height]):
                logging.warning("Invalid YOLO coordinates generated")
                print("Invalid YOLO coordinates generated")
                return None
                
            return [x_center, y_center, width, height]
            
        except Exception as e:
            logging.error(f"Error converting polygon to YOLO format: {str(e)}")
            return None

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
                
            img_width, img_height = img.size
            img.close()
            
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            if not self.validate_ground_truth(gt_data):
                return None
            processed_annotations = []
            processed_annotations = []
            for segment in gt_data['segmentations']['majority-vote']:
                bbox = self.polygon_to_yolo(segment, img_width, img_height, website_id)
                if bbox is not None:
                    processed_annotations.append(bbox)
                    
            return {
                'id': website_id,
                'img_path': img_path,
                'annotations': processed_annotations
            }
            
        except Exception as e:
            logging.error(f"Error processing sample {website_id}: {str(e)}")
            return None

    def visualize_sample(self, sample_id, output_dir=None):
        """Visualize a sample's bounding boxes for debugging"""
        import cv2
        
        if output_dir is None:
            output_dir = self.output_dir / 'debug_visualizations'
            os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(str(sample_id['img_path']))
        height, width = img.shape[:2]
        
        for bbox in sample_id['annotations']:
            # Convert YOLO format to pixel coordinates
            x_center, y_center, w, h = bbox
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        output_path = output_dir / f"{sample_id['id']}_debug.jpg"
        cv2.imwrite(str(output_path), img)
        logging.info(f"Saved visualization for {sample_id['id']} to {output_path}")

    def visualize_small_boxes(self, sample, min_pixels=100):
        """Visualize a sample highlighting very small boxes"""
        import cv2
        output_dir = self.output_dir / 'small_box_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(str(sample['img_path']))
        height, width = img.shape[:2]
        
        has_small_box = False
        for bbox in sample['annotations']:
            x_center, y_center, w, h = bbox
            
            # Convert YOLO format to pixel dimensions
            w_pixels = w * width
            h_pixels = h * height
            area_pixels = w_pixels * h_pixels
            
            # Convert coordinates to pixels
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            
            # Color based on pixel size
            if area_pixels < min_pixels:
                color = (0, 0, 255)  # Red for small boxes
                has_small_box = True
                # Add size annotation
                cv2.putText(img, f'{int(w_pixels)}x{int(h_pixels)}', 
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1)
            else:
                color = (0, 255, 0)  # Green for normal boxes
                
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        if has_small_box:
            output_path = output_dir / f"{sample['id']}_small_boxes.jpg"
            cv2.imwrite(str(output_path), img)
            logging.info(f"Saved small box visualization for {sample['id']}")

    def convert_dataset(self):
        """Convert dataset to YOLO format"""
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
        processed_data = [results_dict[id] for id in sorted(results_dict.keys())]
        # Log statistics about processed data
        logging.info(f"Successfully processed {len(processed_data)} samples out of {len(website_ids)} total")
        
        total_boxes, extremely_small_boxes, box_sizes = self.analyze_box_sizes(processed_data)
        # Calculate and log statistics
        if box_sizes:
            avg_box_size = np.mean(box_sizes) * 100  # Convert to percentage
            print("Avg box size: " + str(avg_box_size))
            median_box_size = np.median(box_sizes) * 100
            print("Median box size: " + str(median_box_size))
            min_box_size = min(box_sizes) * 100 
            print("Min box size: " + str(min_box_size))
            max_box_size = max(box_sizes) * 100
            print("Max box size: " + str(max_box_size))
            
            logging.info(f"""
    Box Statistics:
    - Total boxes: {total_boxes}
    - Small boxes (<0.01% area): {extremely_small_boxes} ({(extremely_small_boxes/total_boxes)*100:.2f}%)
    - Average box size: {avg_box_size:.4f} % 
    - Median box size: {median_box_size:.4f} %
    - Minimum box size: {min_box_size:.4f} %
    - Maximum box size: {max_box_size:.4f} %
    """)

        if len(processed_data) > 0:
            # Visualize first 5 samples for debugging
            for sample in processed_data[:330]:
                self.visualize_sample(sample)
        
        # Split into train and validation sets with fixed ordering
        train_data, val_data = train_test_split(
            processed_data,
            test_size=self.val_split,
            random_state=42,
            shuffle=True  # explicitly state we're shuffling with fixed random state
        )
        
        # Then limit only the training data if num_images is specified
        if self.num_images and self.full_validation_set:
            train_data = train_data[:self.num_images]
        else:
            pass

        logging.info(f"Split dataset into {len(train_data)} train and {len(val_data)} validation samples")
        
        # Save datasets
        for split_name, split_data in [('train', train_data), ('val', val_data)]:
            total_boxes_in_split = 0
            for sample in tqdm(split_data, desc=f"Saving {split_name} split"):
                # Copy image
                shutil.copy(
                    sample['img_path'],
                    self.output_dir / 'images' / split_name / f"{sample['id']}.png"
                )
                
                # Save labels
                label_path = self.output_dir / 'labels' / split_name / f"{sample['id']}.txt"
                with open(label_path, 'w') as f:
                    for bbox in sample['annotations']:
                        f.write(f"0 {' '.join(map(str, bbox))}\n")
                        total_boxes_in_split += 1
            
            logging.info(f"{split_name} split: {len(split_data)} images, {total_boxes_in_split} boxes")
        
        # Create dataset.yaml
        self.create_yaml()
        
        # Final statistics
        logging.info(f"""
    Dataset Summary:
    - Total processed samples: {len(processed_data)}
    - Training samples: {len(train_data)}
    - Validation samples: {len(val_data)}
    - Total bounding boxes: {total_boxes}
    - Average boxes per image: {total_boxes/len(processed_data):.2f}
    """)
        
        return len(processed_data)

    def create_yaml(self):
        """Create YOLO dataset.yaml file"""
        yaml_content = f"""
path: {self.output_dir.absolute()}
train: images/train
val: images/val

names:
  0: webpage_segment
"""
        print(self.output_dir / 'dataset.yaml')
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content.strip())

def main():
    converter = WebisToYOLOConverter(
        source_path="./data/webis-webseg-20/3988124/webis-webseg-20-screenshots/webis-webseg-20",
        ground_truth_path="./data/webis-webseg-20/3988124/webis-webseg-20-ground-truth/webis-webseg-20",
        output_dir="./data/webis-webseg-20-yolo-no-tiny-segments-full",
        num_images=None,  # Process all images
        full_validation_set=True,  # Use full validation set
        val_split=0.2
    )
    
    total_processed = converter.convert_dataset()
    print(f"Conversion completed. Total processed samples: {total_processed}")
main()