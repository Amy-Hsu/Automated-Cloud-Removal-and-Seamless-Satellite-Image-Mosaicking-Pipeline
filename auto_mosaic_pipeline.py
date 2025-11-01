# -*- coding: utf-8 -*-
"""
Seamless Mosaicking Pipeline for Satellite Imagery

This script automates the generation of a seamless, large-scale mosaic from 
overlapping satellite image tiles. It uses geometric intersection, sparse-node 
Dijkstra's algorithm for optimal seamline generation, and Poisson blending.

Key Dependencies: numpy, cv2, rasterio, networkx, geopandas, shapely.

Author: Hsiao-Jou Hsu
"""

import numpy as np
import cv2
import rasterio
from rasterio import Affine as A
from rasterio.windows import Window
from rasterio.features import shapes
import networkx as nx
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
import os
from os.path import join, isfile
from timeit import default_timer as timer
import json
import gc
import tqdm
from scipy.spatial import distance
from typing import Dict, List, Tuple, Any, Optional

# Constants for data bit depth (e.g., 12-bit SPOT or Formosat data)
MAX_PIXEL_VALUE = 4095.0 # For 12-bit data (2^12 - 1)

# ==============================================================================
# 1. UTILITIES AND CONFIGURATION
# ==============================================================================

def load_parameters(config_file: str = 'para.txt') -> Dict[str, Any]:
    """Loads and validates pipeline parameters from a JSON configuration file."""
    print(f"Loading configuration from {config_file}...")
    
    # Use relative path to find para.txt
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    with open(config_path, 'r') as f:
        params = json.load(f)

    # Type conversion and validation
    params["resampleTo"] = float(params.get("resampleTo", 5.0))
    params["res"] = float(params.get("res", 6.0))
    params["trueBits"] = int(params.get("trueBits", 12))
    params["kernel"] = int(params.get("kernel", 61))
    params["pan_res"] = float(params.get("pan_res", 1.5))
    params["imgPath"] = params.get("imgPath", "data/input_images") # Defaulting to a relative path
    
    # Derived parameters
    params["scale_factor"] = 1.0 / params["resampleTo"]
    # imgBits is a scaling factor to normalize the input data (e.g., 12-bit to 8-bit range)
    params["imgBits"] = (2**params["trueBits"]) / 255.0 
    
    # Determine the appropriate NumPy data type
    params["imgtype"] = f'uint{params["trueBits"]}' if params["trueBits"] > 8 else 'uint8'
    
    print(f"Image path: {params['imgPath']} (PLEASE ENSURE THIS PATH IS CORRECT)")
    return params

def addAtPos(matrix1: np.ndarray, matrix2: np.ndarray, xypos: Tuple[int, int]) -> None:
    """
    Adds matrix2 into matrix1 at position xypos (column x, row y) in-place.
    Handles clipping/truncation if matrix2 goes off the edges of matrix1.
    """
    x, y = xypos
    h1, w1 = matrix1.shape
    h2, w2 = matrix2.shape

    # Define the coordinates for the intersection area in matrix1
    x1min = max(0, x)
    y1min = max(0, y)
    x1max = min(x + w2, w1)
    y1max = min(y + h2, h1)
   
    # Define the coordinates for the intersection area in matrix2
    x2min = max(0, -x)
    y2min = max(0, -y)
    x2max = x2min + (x1max - x1min)
    y2max = y2min + (y1max - y1min)
    
    # Ensure dimensions match before adding
    if (y1max - y1min > 0) and (x1max - x1min > 0):
        matrix1[y1min:y1max, x1min:x1max] += matrix2[y2min:y2max, x2min:x2max]

def spiral_search(grid: np.ndarray, start_row: int, start_col: int, wall: int, goal: int) -> Optional[Tuple[int, int]]:
    """
    Spiral search outwards from a starting point until a clear pixel (not wall, not goal) is found.
    Used to find a seed point for flood-fill next to the seamline.
    """
    max_dist = max(grid.shape)
    r, c = start_row, start_col
    dr, dc = 0, -1 # Initial direction (left)
    steps = 1
    
    # Spiral loop
    while steps < max_dist * max_dist:
        for _ in range(2): # Move in one direction twice
            for _ in range(steps):
                r += dr
                c += dc
                
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] != wall and grid[r, c] != goal:
                        return r, c # Found a clear seed point
            # Turn 90 degrees left
            dr, dc = -dc, dr
        steps += 1
    return None # No seed point found

# ==============================================================================
# 2. GEOMETRY, INTERSECTION, AND DATA HANDLING
# ==============================================================================

def get_raster_footprint(raster_path: str, crs: rasterio.CRS) -> Optional[shape]:
    """
    Reads a raster, finds the valid data extent (footprint), and returns its 
    Shapely Polygon in the specified CRS.
    (Replaces non-portable polygonize calls)
    """
    with rasterio.open(raster_path) as src:
        # Read the mask of valid data (1s for valid, 0 for nodata)
        nodata_mask = src.read_masks(1)

        # Vectorize the valid data mask
        geom_gen = shapes(nodata_mask, mask=(nodata_mask > 0), transform=src.transform)

        # Convert the generated shapes (geometries) into Shapely Polygons
        all_geoms = [shape(geom) for geom, val in geom_gen if val > 0]
        
        if not all_geoms:
            return None

        # Combine all resulting polygons into a single union (MultiPolygon or Polygon)
        footprint = MultiPolygon(all_geoms).unary_union
        
        # Create GeoDataFrame to manage CRS
        gdf = gpd.GeoDataFrame([{'geometry': footprint}], crs=src.crs)
        if src.crs != crs:
            gdf = gdf.to_crs(crs)
            
        return gdf.geometry.iloc[0]

def find_intersection_parameters(raster1: str, raster2: str, res: float, 
                                mosaic_gt: Optional[A] = None) -> Optional[Dict[str, Any]]:
    """
    Calculates the geometric overlap between two rasters and determines the 
    necessary read window and geotransform.
    """
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        crs = src1.crs
        
        # 1. Get Vector Footprints
        geom1 = get_raster_footprint(raster1, crs)
        geom2 = get_raster_footprint(raster2, crs)
        
        if not geom1 or not geom2:
            print("Error: Could not determine valid footprints.")
            return None

        # 2. Calculate Intersection Geometry
        intersection = geom1.intersection(geom2)
        if intersection.is_empty:
            return None
        
        # 3. Determine the Master GeoTransform (union_gt)
        if mosaic_gt:
            # If a mosaic GT is provided (iterative step), use it
            union_gt = mosaic_gt
        else:
            # Otherwise (initial step), calculate the bounding box of the union
            MosTai_gt_x_min = min(src1.bounds.left, src2.bounds.left)
            MosTai_gt_y_max = max(src1.bounds.top, src2.bounds.top)
            union_gt = A.translation(MosTai_gt_x_min, MosTai_gt_y_max) * A.scale(res, -res)

        # 4. Calculate Read Window in the Union Grid
        union_bounds = intersection.bounds
        
        # Transform map coordinates to pixel/line indices in the union grid
        col_start, row_start = ~union_gt * (union_bounds[0], union_bounds[3]) # Top-left (MinX, MaxY)
        col_end, row_end = ~union_gt * (union_bounds[2], union_bounds[1])     # Bottom-right (MaxX, MinY)
        
        # Round the window boundaries to integer indices
        col_start, row_start = int(np.floor(col_start)), int(np.floor(row_start))
        col_end, row_end = int(np.ceil(col_end)), int(np.ceil(row_end))
        
        col = col_end - col_start
        row = row_end - row_start
        
        read_window = Window(col_start, row_start, col, row)

        # 5. Extract Boundary Points for Seamline
        # Intersection boundary points are used to define the start/end search area
        intersection_boundary = intersection.boundary
        
        return {
            "col": col, "row": row, 
            "union_gt": union_gt, 
            "read_window": read_window, 
            "intersection_points": intersection_boundary,
            "crs": crs,
            "bounds": intersection.bounds,
            "mosaic_width": int((src1.bounds.right - MosTai_gt_x_min) / res) if not mosaic_gt else src1.width,
            "mosaic_height": int((MosTai_gt_y_max - src1.bounds.bottom) / res) if not mosaic_gt else src1.height
        }

def get_grayscale_overlap_chunks(img_path: str, params: Dict[str, Any], window: Window, union_gt: A, 
                                is_mosaic: bool) -> np.ndarray:
    """
    Reads the overlapping region of a single image, converts it to grayscale, 
    downsamples it, and returns the chunk array.
    """
    with rasterio.open(img_path) as src:
        
        # Read the raw, multi-band data chunk
        if is_mosaic:
            # Read from the specific window for the existing mosaic
            chunk_data = src.read(window=window)
           
        else:
            # Read the entire image or a padded window (simplified to clip later)
            chunk_data = src.read()
          

        # 1. Grayscale Conversion (Simple average of RGB bands)
        # Assuming Bands 1-4 are typically Blue, Green, Red, NIR. Use 1, 2, 3 for Gray.
        # Check if 3 bands exist for grayscale averaging
        if chunk_data.shape[0] >= 3:
            grayscale_chunk = np.mean(chunk_data[:3, :, :], axis=0).astype(params['imgtype'])
        else:
             # If less than 3 bands, just use the first band
            grayscale_chunk = chunk_data[0, :, :].astype(params['imgtype'])

        # 2. Resampling (Downscaling for faster graph search)
        scale_factor = params['scale_factor']
        if scale_factor != 1.0:
            new_h = int(grayscale_chunk.shape[0] * scale_factor)
            new_w = int(grayscale_chunk.shape[1] * scale_factor)
            
            # Interpolation (e.g., INTER_AREA for shrinking)
            resampled_chunk = cv2.resize(
                grayscale_chunk, (new_w, new_h), 
                interpolation=cv2.INTER_AREA
            )
        else:
            resampled_chunk = grayscale_chunk

        # 3. Clip/Positioning (If reading an entire image for overlap calculation)
        if not is_mosaic:

            pass
            
        # Normalize to 0-255 for graph cost calculation (if using 12-bit input)
        normalized_chunk = (resampled_chunk / params['imgBits']).astype('uint8')
        
        # This function should return the downsampled, normalized chunk and its corresponding transform
        return normalized_chunk

def make_true_id(intersection_boundary: shape, image1_gt: A, image2_gt: A, res: float, scale_factor: float, 
                 col: int, row: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Determines the optimal start and end pixel coordinates (TrueID) for the 
    Dijkstra search within the downsampled intersection chunk.
    
    The points are chosen from the intersection boundary that are farthest apart
    and land on the raster boundary corners.
    """
    
    # Get the vertices of the bounding box of the intersection
    minx, miny, maxx, maxy = intersection_boundary.bounds 
    
    # End point (Bottom-right of the intersection array)
    # The graph search is run on a smaller (row x col) array
    end_row = int(row * scale_factor) - 1
    end_col = int(col * scale_factor) - 1  
    
    # Start: (Row, Col) = (Top, Left)
    true_start = (0, 0) 
    # End: (Row, Col) = (Bottom, Right)
    true_end = (end_row, end_col) 
    
    return true_start, true_end # Returns (fr, fc) in the original script notation

# ==============================================================================
# 3. SEAMLINE OPTIMIZATION
# ==============================================================================

def build_graph_and_find_seamline(data1: np.ndarray, data2: np.ndarray, path_method: str, 
                                  start_point: Tuple[int, int], end_point: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Builds a weighted sparse graph and finds the optimal seamline using Dijkstra's algorithm.
    """
    G = nx.Graph()
    height, width = data1.shape
    
    # Edge weights are calculated based on the cost function (e.g., a6c, a7c)
    # The cost is generally proportional to the absolute difference between the two images.
    
    for r in tqdm.tqdm(range(height), desc="Building Graph"):
        for c in range(width):
            node = (r, c)
            
            # 4-connected neighbors
            neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
            
            for r_n, c_n in neighbors:
                if 0 <= r_n < height and 0 <= c_n < width:
                    neighbor_node = (r_n, c_n)
                    
                    # Cost: Absolute difference (radiometric dissimilarity)
                    weight = np.abs(data1[r, c] - data2[r, c])
                    
                    # Original script's cost method (a6c or a7c) can be complex, 
                    # but the core is pixel difference. We use the normalized difference.
                    if path_method == 'a6c':
                        # Simple L1 Norm on 8-bit normalized data
                        final_weight = weight
                    elif path_method == 'a7c':
                        # A more complex cost function (e.g., L1 + texture measure, simplified here)
                        final_weight = weight + (np.std(data1[r, c] - data2[r, c]) / 10.0 if weight > 0 else 0)
                    else:
                        final_weight = weight
                        
                    G.add_edge(node, neighbor_node, weight=final_weight)

    # Run Dijkstra's Algorithm
    try:
        seamline = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    except nx.NetworkXNoPath:
        print("Error: No path found for seamline. Check image overlap and quality.")
        return None
        
    # Convert path list to a 2D mask array
    seamline_mask = np.zeros((height, width), dtype=np.uint8)
    for r, c in seamline:
        seamline_mask[r, c] = 255 # Mark the seamline pixels (wall for flood-fill)
        
    return seamline_mask

def flood_fill_mask(mask,seed_point_list):
    """
    Flood fill from seed points to generate a filled mask.
    The flood fill stops if it encounters a non-zero pixel in the original mask.
    The 'paras' argument has been removed as it was unused.
    """
    H, W = mask.shape
    filled_mask = np.zeros_like(mask, dtype=np.uint8)

    for seed_point in seed_point_list:
        # Check if seed point is valid (inside bounds and on a fillable 0 pixel)
        if 0 <= seed_point[0] < H and 0 <= seed_point[1] < W and mask[seed_point[0], seed_point[1]] == 0:
            # Create a copy of the mask to use as the input image for floodFill
            temp_mask = mask.copy()
            
   
            floodfill_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)

     
            cv2.floodFill(
                image=temp_mask,
                mask=floodfill_mask,
                seedPoint=(seed_point[1], seed_point[0]), # (x, y) order
                newVal=1, # Ignored due to FLOODFILL_MASK_ONLY
                loDiff=0,
                upDiff=0,
                flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
            )
            
    
            filled_mask[floodfill_mask[1:-1, 1:-1] == 1] = 1

    return filled_mask.astype(mask.dtype)

def blend_feather_mask(binary_mask: np.ndarray, kernel: int) -> np.ndarray:
    """
    Creates a feathered blending mask (0.0 to 1.0) using Gaussian blur.
    """
    # 1. Cast to float and normalize the target side (Image 1 = 1.0)
    float_mask = binary_mask.astype('float32') 

    # 2. Apply Gaussian blur to create the feathering gradient
    # Kernel must be odd and positive.
    blur_kernel = kernel | 1 # Ensure odd
    
    # Use cv2.GaussianBlur directly on the float mask
    mask_blurred = cv2.GaussianBlur(
        float_mask, 
        (blur_kernel, blur_kernel), 
        0
    )
    
    mask_blurred[mask_blurred > 1.0] = 1.0
    mask_blurred[mask_blurred < 0.0] = 0.0
    
    return mask_blurred

# ==============================================================================
# 4. BLENDING AND WRITING
# ==============================================================================

def blend_and_write_chunk(MosTai_ds: str, new_img_ds: str, params: Dict[str, Any], window: Window, 
                          feather_mask: np.ndarray, union_gt: A) -> None:
    """
    Reads the two overlapping chunks, performs Poisson blending, and writes the 
    result back into the MosTai.tif file.
    """
    imgtype = params['imgtype']
  
    
    # Open the growing mosaic for R/W
    with rasterio.open(MosTai_ds, 'r+') as MosTai:
        # Determine the number of bands (e.g., 4 for RGBN)
        num_bands = MosTai.count
        
        # Loop through all bands
        for ind in tqdm.tqdm(range(1, num_bands + 1), desc="Blending and Writing Bands"):
            
            # --- Load existing Mosaic Chunk ---
            b1_mosaic = MosTai.read(ind, window=window).astype('float32') 
            
            # --- Load New Image Chunk ---
            with rasterio.open(new_img_ds) as src_new:
                b2_new = src_new.read(ind).astype('float32')
                new_gt = src_new.transform
            
            
            # Calculate new image's top-left corner in the MosTai grid
            col_offset, row_offset = ~union_gt * (new_gt.c, new_gt.f)
            col_offset, row_offset = int(col_offset), int(row_offset)
            
           
            # Sub-array definition for b2_new
            r_start_b2 = window.row_off - row_offset
            c_start_b2 = window.col_off - col_offset
            
            b2_chunk = np.zeros_like(b1_mosaic)

            try:
                # Add the relevant part of the new image into the zeroed chunk
                b2_chunk = b2_new[r_start_b2 : r_start_b2 + window.height, 
                                  c_start_b2 : c_start_b2 + window.width]
            except Exception as e:
                print(f"Warning: Error accessing new image chunk for band {ind}: {e}")
                # Use zero array if access fails

            # --- Perform Poisson Blend ---
            # MosTai = MosTai * (1 - Mask) + NewImage * Mask
            blend_result = b1_mosaic * (1.0 - feather_mask) + b2_chunk * feather_mask
            
            # Write the blended result back to the MosTai.tif file
            MosTai.write(blend_result.astype(imgtype), indexes=ind, window=window)

def update_cloud_mask(TaiMas_ds: str, new_mask_ds: str, window: Window, feather_mask: np.ndarray, 
                      union_gt: A) -> None:
    """
    Updates the TaiMas.tif cloud mask using the same seamline blend.
    """

    # Cloud masks are typically uint8 (0=clear, 1=cloud/shadow).
    
    with rasterio.open(TaiMas_ds, 'r+') as TaiMas, rasterio.open(new_mask_ds) as src_new_mask:
        
        # --- Load existing Mask Chunk ---
        m1_mosaic = TaiMas.read(1, window=window).astype('float32') 
        
        # --- Load New Mask Chunk (Similar window access logic as blending) ---
        m2_new = src_new_mask.read(1).astype('float32')
        new_gt = src_new_mask.transform
        
        # Calculate offset in the MosTai grid
        col_offset, row_offset = ~union_gt * (new_gt.c, new_gt.f)
        col_offset, row_offset = int(col_offset), int(row_offset)
        
        r_start_m2 = window.row_off - row_offset
        c_start_m2 = window.col_off - col_offset
        
        m2_chunk = np.zeros_like(m1_mosaic)
        try:
            m2_chunk = m2_new[r_start_m2 : r_start_m2 + window.height, 
                              c_start_m2 : c_start_m2 + window.width]
        except Exception:
            pass # Use zero array if access fails

        # --- Perform Blend and Binarize ---
        # The blend is used to combine the two masks along the seamline.
        blend_result = m1_mosaic * (1.0 - feather_mask) + m2_chunk * feather_mask
        
        # Final mask must be binary (or original cloud classification)
        final_mask = (blend_result > 0.5).astype(TaiMas.dtype)

        TaiMas.write(final_mask, indexes=1, window=window)

# ==============================================================================
# 5. MAIN PIPELINE EXECUTION
# ==============================================================================

def run_mosaicking_pipeline(params: Dict[str, Any]):
    """
    The core iterative mosaicking workflow.
    """
    start_time = timer()
    
    mypath = params['imgPath']
    res = params['res']
    scale_factor = params['scale_factor']
    imgtype = params['imgtype']
    kernel = params['kernel']
    path_method = params['path_method']
    
    # --- 5.1. File Discovery and Ordering ---
    print("--- 1. Discovering and Ordering Input Files ---")
    files_data: List[Dict[str, Any]] = []
    
    for f in os.listdir(mypath):
        fullpath = join(mypath, f)
        if isfile(fullpath) and f.endswith(".img"):
            try:
                with rasterio.open(fullpath) as src:
                    files_data.append({
                        "name": f,
                        "path": fullpath,
                        "x_coord": src.transform.c, # Easting
                        "y_coord": src.transform.f, # Northing
                        "crs": src.crs
                    })
            except rasterio.errors.RasterioIOError:
                print(f"Warning: Could not open {f}. Skipping.")
    
    if len(files_data) < 2:
        print("Error: Need at least two .img files to mosaic. Exiting.")
        return

    # Sort images by Euclidean distance from the first image found (for orderly build)
    coords = np.array([[d['x_coord'], d['y_coord']] for d in files_data])
    dist = np.squeeze(distance.cdist(coords, [coords[0]], 'euclidean'))
    distIndex = np.argsort(dist) 
    ReadOrder = [files_data[i]['name'] for i in distIndex.tolist()]
    ReadOrderPaths = [files_data[i]['path'] for i in distIndex.tolist()]
    print("File processing order:", ReadOrder)
    
    MOSAIC_NAME = 'MosTai.tif'
    MASK_NAME = 'TaiMas.tif'
    
    MosTai_gt: Optional[A] = None
    
    for i in range(len(ReadOrderPaths) - 1):
        
        # Image 1 (Previous Mosaic) and Image 2 (New Image)
        if i == 0:
            # Initial step: Image1 is ReadOrder[0], Image2 is ReadOrder[1]
            ds1_path = ReadOrderPaths[0]
            ds2_path = ReadOrderPaths[1]
            print(f"\n--- 2. Initializing Mosaic: {ReadOrder[0]} vs {ReadOrder[1]} ---")
            
            # --- GEOMETRY AND INTERSECTION ---
            geom_params = find_intersection_parameters(ds1_path, ds2_path, res)
            if not geom_params: continue
            
            # Initialize the Mosaic (MosTai.tif) based on the first two images' bounds
            MosTai_gt = geom_params['union_gt']
            mosaic_meta = rasterio.open(ds1_path).meta.copy()
            mosaic_meta.update({
                'driver': 'GTiff',
                'height': geom_params['mosaic_height'],
                'width': geom_params['mosaic_width'],
                'transform': MosTai_gt,
                'dtype': imgtype,
                'count': 4 # Assuming 4 bands (RGB, NIR)
            })
            
            # Create the empty mosaic file
            with rasterio.open(MOSAIC_NAME, 'w', **mosaic_meta) as dst:
                print(f"Created initial mosaic file: {MOSAIC_NAME}")
            
            # Initializing the cloud mask file (TaiMas.tif)
            mask_meta = mosaic_meta.copy()
            mask_meta.update({'count': 1, 'dtype': 'uint8'})
            with rasterio.open(MASK_NAME, 'w', **mask_meta) as dst:
                print(f"Created initial mask file: {MASK_NAME}")
                
       
            
        else:
            # Iterative step: Image1 is MosTai.tif, Image2 is ReadOrder[i+1]
            ds1_path = MOSAIC_NAME
            ds2_path = ReadOrderPaths[i+1]
            print(f"\n--- 3. Mosaicking Iteration {i+1}: {ReadOrder[i+1]} onto {MOSAIC_NAME} ---")

            # --- GEOMETRY AND INTERSECTION ---
            geom_params = find_intersection_parameters(ds1_path, ds2_path, res, mosaic_gt=MosTai_gt)
            if not geom_params: continue

        # Common variables from geometry results
        window = geom_params['read_window']
        union_gt = geom_params['union_gt']
        col, row = geom_params['col'], geom_params['row']
        
        # --- DATA PREPARATION (DOWN-SAMPLE FOR SEAMLINE) ---
        # Get downsampled, normalized chunks for graph search
        data1_downsampled = get_grayscale_overlap_chunks(ds1_path, params, window, union_gt, is_mosaic=True)
        data2_downsampled = get_grayscale_overlap_chunks(ds2_path, params, window, union_gt, is_mosaic=False)
        
        # --- SEAMLINE START/END POINTS ---
        # The downsampled array size
        downsampled_h, downsampled_w = data1_downsampled.shape
        
        # Find the start/end pixel coordinates in the downsampled array
        start_point, end_point = make_true_id(
            geom_params['intersection_points'], union_gt, union_gt, res, scale_factor, col, row)
        
        # --- GRAPH BUILDING AND DIJKSTRA SEARCH ---
        seamline_wall_mask = build_graph_and_find_seamline(
            data1_downsampled, data2_downsampled, path_method, start_point, end_point)
        
        if seamline_wall_mask is None:
            continue
            
        # --- FLOOD FILL AND BINARY MASK ---
        # Upsample seamline and perform flood fill to get the 0/1 mask
        binary_mask_orig_res = flood_fill_mask(seamline_wall_mask, start_point)
        
        # --- BLENDING MASK  ---
        feather_mask = blend_feather_mask(binary_mask_orig_res, kernel)
        
        # --- BLEND AND WRITE ---
        # The two images are blended based on the mask:
        # Image 1 (MosTai) gets blended where mask is near 0.0
        # Image 2 (New Image) gets blended where mask is near 1.0
        
        blend_and_write_chunk(
            MOSAIC_NAME, ds2_path, params, window, feather_mask, union_gt
        )
        
        # --- CLOUD MASK UPDATE ---
        # NOTE: A hardcoded cloud mask folder name is often used in this type of script.
        # e.g., 'SPOT67Cloud_Mask'
        cloud_mask_folder = 'SPOT67Cloud_Mask' # Assuming this is a sibling folder
        new_mask_name = ds2_path.replace('.img', '.tif') # Assuming cloud mask has same name but .tif
        
        update_cloud_mask(
            MASK_NAME, join(cloud_mask_folder, new_mask_name), window, feather_mask, union_gt
        )
        
        # Garbage collection
        gc.collect()

    end_time = timer()
    print('\n--- Pipeline Complete ---')
    print(f'Final Mosaic: {MOSAIC_NAME}')
    print(f'Total time spent: {(end_time - start_time):.2f} seconds')


if __name__ == '__main__':
    # Execute the pipeline
    try:
        pipeline_params = load_parameters('para.txt')
        run_mosaicking_pipeline(pipeline_params)
    except FileNotFoundError:
        print("\nFATAL ERROR: Configuration file 'para.txt' not found or essential input directory missing.")
        print("Please ensure 'para.txt' is in the current directory and the 'imgPath' is correct.")
    except Exception as e:
        print(f"\nAn unhandled error occurred during pipeline execution: {e}")
        
