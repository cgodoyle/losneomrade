"""
Compare Serial Performance: Original vs Optimized Landslide Retrogression

This script compares the performance (execution time and result accuracy) of the
original landslide_retrogression function (restored from main branch) vs the
optimized BFS implementation.

It iterates through multiple stream subsets and generates side-by-side plots.

Usage:
    python scripts/compare_serial_performance.py
    python scripts/compare_serial_performance.py --subsets 5 10
"""
import sys
import os
import argparse
import time
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import rasterio
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.abspath("src"))

try:
    from losneomrade import terrain_criteria, retrogression, utils
except ImportError:
    print("Could not import losneomrade. Make sure the 'src' directory is correctly structured.")
    sys.exit(1)


def plot_comparison(ax, dem_file, stream_file, retro_gdf, title, subtitle=""):
    """Plot retrogression results on a given axis"""
    
    # Plot DEM Hillshade
    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(dem_data, vert_exag=1)
        ax.imshow(hs, cmap='gray', extent=extent, origin='upper', alpha=0.8)

    # Plot Streams
    if os.path.exists(stream_file):
        streams = gpd.read_file(stream_file)
        streams.plot(ax=ax, color='blue', linewidth=1.5, zorder=2)

    # Plot Retrogression Results
    if retro_gdf is not None and not retro_gdf.empty:
        retro_gdf.plot(ax=ax, color='red', alpha=0.6, edgecolor='darkred', zorder=4)
        
        n_polygons = len(retro_gdf)
        area_ha = retro_gdf.geometry.area.sum() / 10000
        if not subtitle:
            subtitle = f"{n_polygons} polygons, {area_ha:.2f} ha"
    elif retro_gdf is None:
        subtitle = "No results (Failed)"
    else:
        subtitle = "No results (Empty)"

    # Formatting
    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Streams'),
        Patch(facecolor='red', edgecolor='darkred', alpha=0.6, label='Retrogression')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)


def run_comparison(subsets, dem_file, output_dir):
    print("\n" + "="*80)
    print("COMPARING SERIAL PERFORMANCE: ORIGINAL VS OPTIMIZED")
    print("="*80)
    
    # Parameters
    SOURCE_DEPTH = 0.0
    MIN_HEIGHT = 5.0
    MIN_SLOPE = 1 / 15
    MIN_LENGTH = 75.0
    POINTS_PER_METER = 1 / 10
    INITIAL_BUFFER = 10
    BUFFER_DISTANCE = 300
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    summary_data = []

    for subset in subsets:
        stream_file = f"data/streams_subset_{subset}.geojson"
        if not os.path.exists(stream_file):
            print(f"\n[SKIP] Stream file not found: {stream_file}")
            continue
            
        print(f"\n" + "-"*60)
        print(f"Processing Subset: {subset}")
        print("-"*-60)
        
        # Load streams
        streams = gpd.read_file(stream_file)
        print(f"Loaded {len(streams)} stream features")

        # Calculate bounds
        bounds_array = streams.total_bounds
        bounds = (
            bounds_array[0] - BUFFER_DISTANCE,
            bounds_array[2] + BUFFER_DISTANCE,
            bounds_array[1] - BUFFER_DISTANCE,
            bounds_array[3] + BUFFER_DISTANCE,
        )

        # Generate source points
        try:
            source_points = terrain_criteria.generate_source_points(
                streams,
                distance_chainage=1 / POINTS_PER_METER,
            )
        except Exception as e:
            print(f"[ERROR] generating source points: {e}")
            continue

        # Create initial release zones
        try:
            point_geometries = [Point(x, y) for x, y in source_points[:, :2]]
            points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=streams.crs)
            buffered = points_gdf.buffer(INITIAL_BUFFER)
            buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=streams.crs)
            initial_release = buffered_gdf.dissolve()
        except Exception as e:
            print(f"[ERROR] creating initial release zones: {e}")
            continue

        # Prepare DEM
        if dem_file is None:
            dem_data = utils.get_hoydedata(bounds)
        else:
            dem_data = utils.generate_windows(dem_file)
            
        dem_array = dem_data["full_array"]
        dem_profile = dem_data["profile"]
        dem_transform = dem_profile["transform"]
        
        # Rasterize release
        rel = utils.rasterize_shape(initial_release, dem_profile)
        
        # Common arguments
        kwargs = {
            "dem": dem_array,
            "initial_release": rel,
            "dem_transform": dem_transform,
            "initial_release_depth": SOURCE_DEPTH,
            "min_slope": MIN_SLOPE,
            "min_height": MIN_HEIGHT,
            "min_length": MIN_LENGTH,
            "mask": None,
            "verbose": True
        }
        
        # 1. Run Original
        print("Running ORIGINAL implementation...")
        start_orig = time.time()
        try:
            release_orig, _ = retrogression.landslide_retrogression_original(**kwargs)
            time_orig = time.time() - start_orig
            print(f"[OK] Completed in {time_orig:.4f}s")
        except AttributeError:
            print("[ERROR] landslide_retrogression_original not found!")
            continue
        except Exception as e:
            print(f"[ERROR] Original failed: {e}")
            time_orig = None
            release_orig = None

        # 2. Run Optimized
        print("Running OPTIMIZED implementation...")
        start_opt = time.time()
        try:
            release_opt, _ = retrogression.landslide_retrogression_optimized(**kwargs)
            time_opt = time.time() - start_opt
            print(f"[OK] Completed in {time_opt:.4f}s")
        except Exception as e:
            print(f"[ERROR] Optimized failed: {e}")
            time_opt = None
            release_opt = None

        # Comparison & Saving
        if release_orig is not None and release_opt is not None:
            # Metrics
            area_orig = np.sum(release_orig)
            area_opt = np.sum(release_opt)
            intersection = np.sum(release_orig & release_opt)
            union = np.sum(release_orig | release_opt)
            iou = intersection / union if union > 0 else 1.0
            speedup = time_orig / time_opt if time_opt > 0 else 0
            
            print(f"Speedup: {speedup:.2f}x")
            print(f"IoU: {iou:.4f}")
            
            summary_data.append({
                "subset": subset,
                "time_original": time_orig,
                "time_optimized": time_opt,
                "speedup": speedup,
                "area_original": area_orig,
                "area_optimized": area_opt,
                "iou": iou
            })
            
            # Incremental Save
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
            print(f"Saved metrics to {os.path.join(output_dir, 'summary_metrics.csv')}")
            
            # Save GeoJSONs
            gdf_orig = utils.polygonize_results(release_orig, dem_profile, field="slope").to_crs(epsg=25833)
            gdf_opt = utils.polygonize_results(release_opt, dem_profile, field="slope").to_crs(epsg=25833)
            
            gdf_orig.to_file(os.path.join(output_dir, f"serial_result_original_{subset}.geojson"), driver="GeoJSON")
            gdf_opt.to_file(os.path.join(output_dir, f"serial_result_optimized_{subset}.geojson"), driver="GeoJSON")
            
            # Generate Plot
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 10))
                
                plot_comparison(axes[0], dem_file, stream_file, gdf_orig, 
                              "Serial Original", f"Time: {time_orig:.2f}s")
                
                plot_comparison(axes[1], dem_file, stream_file, gdf_opt,
                              "Serial Optimized", f"Time: {time_opt:.2f}s | Speedup: {speedup:.2f}x")
                
                fig.suptitle(f"Serial Performance Comparison - Subset {subset}\nIoU: {iou:.4f}", 
                            fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.05)
                
                viz_output = os.path.join(output_dir, f"serial_comparison_{subset}.png")
                plt.savefig(viz_output, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot to {viz_output}")
            except Exception as e:
                print(f"[ERROR] Plotting failed: {e}")

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
        print(f"\nSaved summary metrics to {os.path.join(output_dir, 'summary_metrics.csv')}")
    else:
        print("No successful comparisons.")


def main():
    parser = argparse.ArgumentParser(description="Compare Serial Performance")
    parser.add_argument("--subsets", nargs="+", type=int, default=[1, 5, 10, 15, 25, 50, 100], 
                        help="List of subsets to process (default: 1 5 10 15 25 50 100)")
    parser.add_argument("--dem-file", default="data/dem_byneset_5m.tif", help="Path to DEM file")
    parser.add_argument("--output-dir", default="output/comparison_landslide_retrogression_original_v_optimized", 
                        help="Directory to save results")
    args = parser.parse_args()
    
    run_comparison(args.subsets, args.dem_file, args.output_dir)


if __name__ == "__main__":
    main()
