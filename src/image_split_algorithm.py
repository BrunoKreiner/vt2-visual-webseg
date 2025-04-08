import cv2
import numpy as np


def split_tall_image(image_path, min_height_width_ratio=2.0):
    """
    Split a tall image into two parts if:
    1. Height is at least double the width
    2. We can find a horizontal line that doesn't cross any vertical edges
    
    Args:
        image_path: Path to the image file
        min_height_width_ratio: Minimum height/width ratio to consider splitting
        
    Returns:
        If split is possible: List of two image arrays
        If split is not possible: Original image in a list
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return [None]
    
    height, width = image.shape[:2]
    
    # Check if image is tall enough to consider splitting
    if height / width < min_height_width_ratio:
        return [image]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find vertical edges
    # Use morphology to strengthen vertical edges
    kernel = np.ones((15, 1), np.uint8)  # Vertical kernel
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Create a histogram of horizontal lines where we can split
    # For each row, count the number of vertical edge pixels
    horizontal_histogram = np.sum(vertical_edges, axis=1)
    
    # Find regions with zero vertical edges - these are potential split points
    potential_splits = []
    for y in range(height // 4, 3 * height // 4):  # Look in middle half for balance
        # Check if this row has no vertical edges
        if horizontal_histogram[y] == 0:
            # Check a few rows around to ensure we have a clean split zone
            zone_size = 10
            zone_start = max(0, y - zone_size // 2)
            zone_end = min(height, y + zone_size // 2)
            zone_sum = np.sum(horizontal_histogram[zone_start:zone_end])
            
            if zone_sum == 0:
                potential_splits.append(y)
    
    # If we found potential split points, use the one closest to the middle
    if potential_splits:
        middle = height // 2
        split_point = min(potential_splits, key=lambda y: abs(y - middle))
        
        # Split the image
        upper_half = image[:split_point]
        lower_half = image[split_point:]
        
        return [upper_half, lower_half]
    
    # No suitable split point found
    return [image]