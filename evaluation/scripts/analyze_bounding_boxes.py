#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script analyzes bounding box distributions and calculates statistics for two preprocessing methods:
1. webis-webseg-20-yolo-no-tiny-segments-full
2. IIS_data_yolo_annotator_2_nano

It calculates the average number of bounding boxes per image and creates visualizations
of the bounding box area distributions for both preprocessing methods.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path

# Define the paths to the datasets
DATASET1_PATH = "../../data/data/webis-webseg-20-yolo-no-tiny-segments-full"
DATASET2_PATH = "../../data/data/IIS_data_yolo_annotator_2_nano"

# Define output directory for saving results
OUTPUT_DIR = "../"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_labels(labels_path, images_path):
    """
    Process YOLO format labels and calculate bounding box areas
    
    Args:
        labels_path: Path to the labels directory
        images_path: Path to the images directory
    
    Returns:
        tuple: (bbox_areas, bbox_counts, avg_bbox_count)
    """
    bbox_areas = []
    bbox_counts = []
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_path, "*.txt"))
    
    for label_file in tqdm(label_files, desc="Processing labels"):
        # Get corresponding image file
        img_file = os.path.join(images_path, os.path.basename(label_file).replace(".txt", ".png"))
        
        # Skip if image doesn't exist
        if not os.path.exists(img_file):
            continue
        
        # Get image dimensions
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Read bounding boxes from label file
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            # Count bounding boxes in this image
            bbox_counts.append(len(lines))
            
            # Process each bounding box
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO format: class x_center y_center width height
                    # Convert YOLO format (normalized) to pixel values
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Calculate area
                    area = width * height
                    bbox_areas.append(area)
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
            continue
    
    # Calculate average number of bounding boxes per image
    avg_bbox_count = np.mean(bbox_counts) if bbox_counts else 0
    
    return bbox_areas, bbox_counts, avg_bbox_count

def plot_bbox_area_distribution(bbox_areas, title, output_path):
    """
    Plot the distribution of bounding box areas
    
    Args:
        bbox_areas: List of bounding box areas
        title: Title for the plot
        output_path: Path to save the plot
    """
    plt.style.use('bmh')
    fig = plt.figure(figsize=(10, 6))
    
    # Plot bounding box area distribution
    plt.hist(bbox_areas, bins=50, edgecolor='black')
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Count')
    
    # Remove grid lines
    plt.gca().grid(False)
    
    # Calculate statistics
    avg_area = np.mean(bbox_areas)
    min_area = np.min(bbox_areas)
    max_area = np.max(bbox_areas)
    median_area = np.median(bbox_areas)
    
    # Add statistics text at the bottom of the plot
    stats_text = f'Average area: {avg_area:.2f} pixels²    Minimum area: {min_area:.2f} pixels²\nMaximum area: {max_area:.2f} pixels²    Median area: {median_area:.2f} pixels²'
    plt.figtext(0.5, 0.01, stats_text, fontsize=10, ha='center')
    
    plt.tight_layout()
    # Adjust bottom margin to make room for the text
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    print("Analyzing Dataset 1: webis-webseg-20-yolo-no-tiny-segments-full")
    # Process Dataset 1
    dataset1_train_labels = os.path.join(DATASET1_PATH, "labels", "train")
    dataset1_train_images = os.path.join(DATASET1_PATH, "images", "train")
    dataset1_val_labels = os.path.join(DATASET1_PATH, "labels", "val")
    dataset1_val_images = os.path.join(DATASET1_PATH, "images", "val")
    
    # Process training set
    print("Processing training set...")
    train_bbox_areas1, train_bbox_counts1, train_avg_bbox_count1 = process_labels(dataset1_train_labels, dataset1_train_images)
    
    # Process validation set
    print("Processing validation set...")
    val_bbox_areas1, val_bbox_counts1, val_avg_bbox_count1 = process_labels(dataset1_val_labels, dataset1_val_images)
    
    # Combine train and val for overall statistics
    bbox_areas1 = train_bbox_areas1 + val_bbox_areas1
    bbox_counts1 = train_bbox_counts1 + val_bbox_counts1
    avg_bbox_count1 = np.mean(bbox_counts1) if bbox_counts1 else 0
    
    print(f"Dataset 1 - Average bounding boxes per image: {avg_bbox_count1:.2f}")
    print(f"Dataset 1 - Total number of bounding boxes: {len(bbox_areas1)}")
    print(f"Dataset 1 - Total number of images: {len(bbox_counts1)}")
    
    # Plot distribution for Dataset 1
    plot_bbox_area_distribution(
        bbox_areas1,
        "Bounding Box Area Distribution - webis-webseg-20-yolo-no-tiny-segments",
        os.path.join(OUTPUT_DIR, "bbox_distribution_dataset1.png")
    )
    
    print("\nAnalyzing Dataset 2: IIS_data_yolo_annotator_2_nano")
    # Process Dataset 2
    dataset2_train_labels = os.path.join(DATASET2_PATH, "labels", "train")
    dataset2_train_images = os.path.join(DATASET2_PATH, "images", "train")
    dataset2_val_labels = os.path.join(DATASET2_PATH, "labels", "val")
    dataset2_val_images = os.path.join(DATASET2_PATH, "images", "val")
    
    # Process training set
    print("Processing training set...")
    train_bbox_areas2, train_bbox_counts2, train_avg_bbox_count2 = process_labels(dataset2_train_labels, dataset2_train_images)
    
    # Process validation set
    print("Processing validation set...")
    val_bbox_areas2, val_bbox_counts2, val_avg_bbox_count2 = process_labels(dataset2_val_labels, dataset2_val_images)
    
    # Combine train and val for overall statistics
    bbox_areas2 = train_bbox_areas2 + val_bbox_areas2
    bbox_counts2 = train_bbox_counts2 + val_bbox_counts2
    avg_bbox_count2 = np.mean(bbox_counts2) if bbox_counts2 else 0
    
    print(f"Dataset 2 - Average bounding boxes per image: {avg_bbox_count2:.2f}")
    print(f"Dataset 2 - Total number of bounding boxes: {len(bbox_areas2)}")
    print(f"Dataset 2 - Total number of images: {len(bbox_counts2)}")
    
    # Plot distribution for Dataset 2
    plot_bbox_area_distribution(
        bbox_areas2,
        "Bounding Box Area Distribution - IIS_data_yolo_annotator_2_nano",
        os.path.join(OUTPUT_DIR, "bbox_distribution_dataset2.png")
    )
    
    # Create a combined plot for comparison
    plt.style.use('bmh')
    fig = plt.figure(figsize=(12, 7))
    
    # Plot histograms
    plt.hist(bbox_areas1, bins=50, alpha=0.5, label='webis-webseg-20-yolo-no-tiny-segments')
    plt.hist(bbox_areas2, bins=50, alpha=0.5, label='IIS_data_yolo_annotator_2_nano')
    
    plt.title('Comparison of Bounding Box Area Distributions', fontsize=14, pad=20)
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Count')
    plt.legend()
    
    # Remove grid lines
    plt.gca().grid(False)
    
    # Add statistics text at the bottom of the plot
    stats_text = (
        f'webis-webseg-20: Avg boxes/image: {avg_bbox_count1:.2f}, Avg area: {np.mean(bbox_areas1):.2f} pixels²\n'
        f'IIS_data_nano: Avg boxes/image: {avg_bbox_count2:.2f}, Avg area: {np.mean(bbox_areas2):.2f} pixels²'
    )
    plt.figtext(0.5, 0.01, stats_text, fontsize=10, ha='center')
    
    plt.tight_layout()
    # Adjust bottom margin to make room for the text
    plt.subplots_adjust(bottom=0.15)
    
    # Save the combined plot
    plt.savefig(os.path.join(OUTPUT_DIR, "bbox_distribution_comparison.png"))
    plt.close()
    
    # Save statistics to a text file
    with open(os.path.join(OUTPUT_DIR, "bbox_statistics.txt"), 'w') as f:
        f.write("Bounding Box Statistics\n")
        f.write("======================\n\n")
        
        f.write("Dataset 1: webis-webseg-20-yolo-no-tiny-segments-full\n")
        f.write(f"Average bounding boxes per image: {avg_bbox_count1:.2f}\n")
        f.write(f"Total number of bounding boxes: {len(bbox_areas1)}\n")
        f.write(f"Total number of images: {len(bbox_counts1)}\n")
        f.write(f"Average bounding box area: {np.mean(bbox_areas1):.2f} pixels²\n")
        f.write(f"Minimum bounding box area: {np.min(bbox_areas1):.2f} pixels²\n")
        f.write(f"Maximum bounding box area: {np.max(bbox_areas1):.2f} pixels²\n")
        f.write(f"Median bounding box area: {np.median(bbox_areas1):.2f} pixels²\n\n")
        
        f.write("Dataset 2: IIS_data_yolo_annotator_2_nano\n")
        f.write(f"Average bounding boxes per image: {avg_bbox_count2:.2f}\n")
        f.write(f"Total number of bounding boxes: {len(bbox_areas2)}\n")
        f.write(f"Total number of images: {len(bbox_counts2)}\n")
        f.write(f"Average bounding box area: {np.mean(bbox_areas2):.2f} pixels²\n")
        f.write(f"Minimum bounding box area: {np.min(bbox_areas2):.2f} pixels²\n")
        f.write(f"Maximum bounding box area: {np.max(bbox_areas2):.2f} pixels²\n")
        f.write(f"Median bounding box area: {np.median(bbox_areas2):.2f} pixels²\n")
    
    print("\nAnalysis complete. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
