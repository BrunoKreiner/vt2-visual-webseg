
import numpy as np
from shapely import GeometryCollection
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid, explain_validity
from src.canny import apply_canny, count_edges_in_polygon


def pixel_based_clusterings(segmentations, path):
    # Convert segmentations to a GeometryCollection        
    segmentations.geometrics = as_geometric_collection(segmentations) 
    # segmentations.geometrics = make_valid(segmentations.geometrics)
    
    geom = segmentations.geometrics
    if not geom.is_valid:
        print(f"Invalid geometry: {explain_validity(geom)}")

        geom = geom.buffer(0)  # Attempt to fix the geometry
        
        if not geom.is_valid:
            raise ValueError(f"Could not fix geometry: {explain_validity(geom)}")
        
    # Split the overlapping polygons into distinct segments
    distinct_segment = split_polygons(segmentations.geometrics.geoms)

    # Build membership matrix
    membership = {}
    membership['matrix'] = np.zeros((len(distinct_segment), len(segmentations.geometrics.geoms)), dtype=int)

    for i, distinct_poly in enumerate(distinct_segment):
        for j, original_poly in enumerate(segmentations.geometrics.geoms):

            intersects = distinct_poly.intersection(original_poly)
            if intersects.geom_type == 'Polygon' and not intersects.is_empty and intersects.area > 0:
                membership['matrix'][i, j] = 1

    # Define subsets for each annotator
    membership_subsets = get_subsets(segmentations)  # Placeholder for subset logic

    clusterings = {'clusters': [], 
                   'membership': membership_subsets}
    
    # get edges with canny
    edges_fine = apply_canny(path, 'fine')
    edges_coarse = apply_canny(path, 'coarse')

    for i, seg in enumerate(distinct_segment):
        info = {
            'polygon': seg, 
            'membership': membership['matrix'][i],
            'size_edges_fine': size_function(seg, 'edges', edges_fine),
            'size_edges_coarse': size_function(seg, 'edges', edges_coarse),
            'size_pixel': size_function(seg, 'pixel', None)
        }
        clusterings['clusters'].append(info)
    
    return clusterings


def as_geometric_collection(segmentations):
    polygons = [polygon for value in segmentations.values() for polygon in value]
    # Remove duplicate polygons
    # unique_polygons = []
    # for polygon in polygons:
    #     if not any(polygon.equals(existing_polygon) for existing_polygon in unique_polygons):
    #         unique_polygons.append(polygon)
        
    gc = GeometryCollection(polygons)
    return gc

def get_subsets(segmentations):
    dict_segmens = {polygon:i for i, polygon in enumerate(segmentations.geometrics.geoms)}
    subsets = []
    for key, annot_list in segmentations:
        subsets_annot = []
        for annot in annot_list:
            subsets_annot.append(dict_segmens[annot])
        subsets.append(subsets_annot)
    return subsets

def split_polygons(polygons):
    lines = [LineString(pol.exterior) for pol in polygons]        
    union = unary_union(lines)
    result = [geom for geom in polygonize(union)]
    
    return result

def size_function(seg, unit, edges):

    if unit == 'pixel':
        return seg.area
    if unit == 'edges':
        edges = count_edges_in_polygon(seg, edges)
        if edges == 'error':
            print('ERROR: polygon not on image. (return None)')
            return None
        else:
            return edges
    else:
        print(f'Size function: {unit} not implemented.')
        exit()

