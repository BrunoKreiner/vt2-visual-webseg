import os
import subprocess
import sys

import cv2
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from shapely.geometry import mapping


def apply_canny(input_path,canny_type):
    # print('Applying Canny edge detection...')
    
    if canny_type == 'fine':
        upper_percent = 2
        sigma = 1
    elif canny_type == 'coarse':
        upper_percent = 16
        sigma = 5
    else:
        print(f'Error: Invalid type {canny_type} for applying canny')
        sys.exit(1)
    
    radius = 0
    lower_percent = 1
    
    output_filename = f"screenshot-canny-{radius}x{sigma}-{lower_percent}-{upper_percent}.png"
    output_path = os.path.join(input_path, output_filename)
    input_path = input_path+'screenshot.png'
    
    if not os.path.exists(output_path):
        try:
            if not os.path.exists(input_path):
                print(f"Input path doesn't exists: {input_path}")
            subprocess.check_call([
                'convert', input_path, 
                '-canny', f'{radius}x{sigma}+{lower_percent}%+{upper_percent}%', 
                output_path
            ])
        except subprocess.CalledProcessError as e:
            print('error:', e)
            print("Failed")
            sys.exit(1)
        
    return cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)


def count_edges_in_polygon(polygon,edges, buffer_default = 1):

    # Reverse the y-coordinates (flip vertically) to match ImageMagick behavior
    mask_egdes = np.flipud(edges)
    
    # Create an in-memory raster from the mask using rasterio
    dimensions = mask_egdes.shape
    transform = from_origin(0, dimensions[0], 1, 1)  # Assume a simple 1x1 resolution for each pixel
    
    # Create metadata for the raster
    metadata = {
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',  # Assuming a lat/lon projection for simplicity
        'width': dimensions[1],
        'height': dimensions[0],
        'transform': transform
    }
    
    # Save the mask to an in-memory raster file using rasterio
    with rasterio.open('/vsimem/mask.tif', 'w', **metadata) as dst:
        dst.write(mask_egdes, 1)

    # Enlarging the multipolygon using the buffer
    buffered_polygon = polygon.buffer(buffer_default)
    # Convert the buffered polygon to geojson format
    geojson = [mapping(buffered_polygon)]    
            
    # Open the raster and extract the values within the buffered polygon
    try:
        with rasterio.open('/vsimem/mask.tif') as src:
            out_image, out_transform = mask(src, geojson, crop=True)
    except:
        return 'error'

    # Return the sum of the values in the polygon region
    return np.sum(out_image)
