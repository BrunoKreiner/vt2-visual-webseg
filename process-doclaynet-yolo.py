import json
import os
from pathlib import Path
import shutil

def convert_coco_to_yolo(coco_path, images_path, output_path):
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels'), exist_ok=True)
    
    # Load COCO format annotations
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
    
    # Process each image
    for img in coco_data['images']:
        img_id = img['id']
        img_width = img['width']
        img_height = img['height']
        
        # Copy image
        src_img_path = os.path.join(images_path, img['file_name'])
        dst_img_path = os.path.join(output_path, 'images', img['file_name'])
        shutil.copy2(src_img_path, dst_img_path)
        
        # Convert annotations to YOLO format
        if img_id in image_to_anns:
            label_path = os.path.join(output_path, 'labels', 
                                    os.path.splitext(img['file_name'])[0] + '.txt')
            
            with open(label_path, 'w') as f:
                for ann in image_to_anns[img_id]:
                    # Get category id (subtract 1 to make zero-based)
                    category_id = ann['category_id'] - 1
                    
                    # Get bbox coordinates (COCO format: x,y,width,height)
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format (center_x, center_y, width, height, normalized)
                    center_x = (x + w/2) / img_width
                    center_y = (y + h/2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Write to file
                    f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

# Create data.yaml
def create_data_yaml(output_path):
    yaml_content = """
train: ./images/train
val: ./images/val
test: ./images/test

nc: 11
names: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
    """
    
    with open(os.path.join(output_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content.strip())

def main():
    base_path = './data/DocLayNet_core'
    output_base = './data/DocLayNet_yolo'
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        output_path = os.path.join(output_base, split)
        coco_path = os.path.join(base_path, 'COCO', f'{split}.json')
        images_path = os.path.join(base_path, 'PNG')
        
        print(f"Converting {split} split...")
        convert_coco_to_yolo(coco_path, images_path, output_path)
    
    # Create data.yaml
    create_data_yaml(output_base)
    print("Conversion complete!")

if __name__ == '__main__':
    main()