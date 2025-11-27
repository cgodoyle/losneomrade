import sys
import os
import time
import tracemalloc
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import distance_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from losneomrade import utils, retrogression, terrain_criteria

def benchmark_function(func, args, kwargs, name="Function"):
    print(f"\n--- Benchmarking: {name} ---")
    
    # Measure memory
    tracemalloc.start()
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    peak_mb = peak / 10**6
    
    print(f"Duration: {duration:.4f} seconds")
    print(f"Peak Memory: {peak_mb:.2f} MB")
    
    return result

def run_benchmark():
    print("======================================================================")
    print("PERFORMANCE BENCHMARK")
    print("======================================================================")
    
    # Setup Data
    print("Setting up test data...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    stream_file = os.path.join(base_dir, 'data', 'streams_subset.geojson')
    dem_file = os.path.join(base_dir, 'data', 'dem_byneset_5m.tif')
    
    if not os.path.exists(stream_file):
        print(f"Error: Stream file not found: {stream_file}")
        return
        
    streams = gpd.read_file(stream_file)
    bounds = streams.total_bounds
    # Expand bounds slightly
    bounds = (bounds[0]-100, bounds[1]-100, bounds[2]+100, bounds[3]+100)
    
    # Load DEM window
    window_data = utils.generate_windows(dem_file)
    # Take the first window (assuming subset fits in one or we just use the first for testing)
    # Actually, generate_windows returns a list of windows.
    # For the benchmark, let's just use the full array if it's small enough, or a crop.
    dem_array = window_data['full_array']
    dem_transform = window_data['profile']['transform']
    coords = utils.dem_coordinates(dem_array, dem_transform)
    
    # Generate dummy source points
    # Create a grid of points to simulate a heavy load
    # 1000 points for quick test, 5000 for stress test
    n_points = 25
    print(f"Generating {n_points} random source points...")
    
    # Random points within bounds
    xmin, ymin, xmax, ymax = bounds
    x_rand = np.random.uniform(xmin, xmax, n_points)
    y_rand = np.random.uniform(ymin, ymax, n_points)
    z_rand = np.zeros(n_points) # Dummy Z
    points = np.c_[x_rand, y_rand, z_rand]
    
    # Update Z from raster
    points = utils.set_z_from_raster(points, window_data)
    
    # ---------------------------------------------------------
    # 1. Benchmark Terrain Criteria (compute_slope)
    # ---------------------------------------------------------
    print("\n[Test 1: Terrain Criteria Memory Optimization]")
    
    # Baseline
    print("Running Baseline (compute_slope)...")
    # Note: compute_slope takes (coords, points)
    # We expect high memory usage here
    benchmark_function(utils.compute_slope, (coords, points), {}, name="Baseline: compute_slope")
    
    # Optimized
    print("Running Optimized (compute_slope_chunked)...")
    benchmark_function(utils.compute_slope_chunked, (coords, points), {'chunk_size': 10}, name="Optimized: compute_slope_chunked")
    
    # ---------------------------------------------------------
    # 2. Benchmark Retrogression (Parallel)
    # ---------------------------------------------------------
    print("\n[Test 2: Retrogression Runtime Optimization]")
    
    # Create a dummy release shape (buffered points)
    print("Creating dummy release zones...")
    # Take 5 random points to create 5 independent zones
    subset_points = points[:5]
    point_geoms = gpd.points_from_xy(subset_points[:,0], subset_points[:,1])
    initial_release = gpd.GeoDataFrame(geometry=point_geoms, crs=25833).buffer(20) # 20m radius
    initial_release_gdf = gpd.GeoDataFrame(geometry=initial_release, crs=25833)
    
    # Baseline
    print("Running Baseline (run_retrogression)...")
    # Use verbose=False to keep output clean
    benchmark_function(retrogression.run_retrogression, 
                      (bounds, initial_release_gdf), 
                      {'custom_raster': dem_file, 'verbose': False, 'min_length': 50}, 
                      name="Baseline: run_retrogression")
                      
    # Optimized
    print("Running Optimized (run_retrogression_parallel)...")
    benchmark_function(retrogression.run_retrogression_parallel, 
                      (bounds, initial_release_gdf), 
                      {'custom_raster': dem_file, 'min_length': 50, 'n_processes': 2}, 
                      name="Optimized: run_retrogression_parallel")

if __name__ == "__main__":
    run_benchmark()
