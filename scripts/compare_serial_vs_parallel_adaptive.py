"""
Compare Performance: Serial Optimized vs Parallel Adaptive (Speed)

This script compares the performance (execution time and result accuracy) of the
Serial Optimized implementation vs the Parallel Adaptive implementation (Speed mode).

It iterates through multiple stream subsets and generates side-by-side plots.

Usage:
    python scripts/compare_serial_vs_parallel_adaptive.py
    python scripts/compare_serial_vs_parallel_adaptive.py --subsets 5 10
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
    print("COMPARING PERFORMANCE: SERIAL OPTIMIZED VS PARALLEL ADAPTIVE (SPEED)")
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

        # Common arguments for wrappers
        kwargs = {
            "bounds": bounds,
            "rel_shape": initial_release,
            "point_depth": SOURCE_DEPTH,
            "clip_to_msml": False,
            "min_slope": MIN_SLOPE,
            "min_height": MIN_HEIGHT,
            "min_length": MIN_LENGTH,
            "custom_raster": dem_file,
            "verbose": True
        }
        
        # 1. Run Serial Optimized
        print("Running SERIAL OPTIMIZED implementation...")
        start_serial = time.time()
        try:
            # Using run_retrogression which wraps landslide_retrogression_optimized
            gdf_serial = retrogression.run_retrogression(
                **kwargs,
                return_animation=False
            )
            time_serial = time.time() - start_serial
            print(f"[OK] Completed in {time_serial:.4f}s")
        except Exception as e:
            print(f"[ERROR] Serial Optimized failed: {e}")
            import traceback
            traceback.print_exc()
            time_serial = None
            gdf_serial = None

        # 2. Run Parallel Adaptive (Speed)
        print("Running PARALLEL ADAPTIVE (SPEED) implementation...")
        
        # Remove verbose from kwargs for parallel function as it doesn't support it
        kwargs_parallel = kwargs.copy()
        if "verbose" in kwargs_parallel:
            del kwargs_parallel["verbose"]
            
        start_parallel = time.time()
        try:
            gdf_parallel = retrogression.run_retrogression_parallel_adaptive(
                **kwargs_parallel,
                speed_priority='speed'
            )
            time_parallel = time.time() - start_parallel
            print(f"[OK] Completed in {time_parallel:.4f}s")
        except Exception as e:
            print(f"[ERROR] Parallel Adaptive failed: {e}")
            import traceback
            traceback.print_exc()
            time_parallel = None
            gdf_parallel = None

        # Comparison & Saving
        if gdf_serial is not None and gdf_parallel is not None:
            # Metrics
            area_serial = gdf_serial.geometry.area.sum()
            area_parallel = gdf_parallel.geometry.area.sum()
            
            # Calculate IoU (approximate using geometry union/intersection)
            try:
                geom_serial = gdf_serial.union_all()
                geom_parallel = gdf_parallel.union_all()
                
                if geom_serial.is_empty and geom_parallel.is_empty:
                    iou = 1.0
                elif geom_serial.is_empty or geom_parallel.is_empty:
                    iou = 0.0
                else:
                    intersection = geom_serial.intersection(geom_parallel).area
                    union = geom_serial.union(geom_parallel).area
                    iou = intersection / union if union > 0 else 0.0
            except Exception as e:
                print(f"[WARN] IoU calculation failed: {e}")
                iou = 0.0

            speedup = time_serial / time_parallel if time_parallel > 0 else 0
            
            print(f"Speedup: {speedup:.2f}x")
            print(f"IoU: {iou:.4f}")
            
            summary_data.append({
                "subset": subset,
                "time_serial_optimized": time_serial,
                "time_parallel_adaptive": time_parallel,
                "speedup": speedup,
                "area_serial_optimized": area_serial,
                "area_parallel_adaptive": area_parallel,
                "iou": iou
            })
            
            # Incremental Save
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
            print(f"Saved metrics to {os.path.join(output_dir, 'summary_metrics.csv')}")
            
            # Save GeoJSONs
            gdf_serial.to_file(os.path.join(output_dir, f"serial_optimized_result_{subset}.geojson"), driver="GeoJSON")
            gdf_parallel.to_file(os.path.join(output_dir, f"parallel_adaptive_speed_result_{subset}.geojson"), driver="GeoJSON")
            
            # Generate Plot
            try:
                fig, axes = plt.subplots(1, 2, figsize=(16, 10))
                
                plot_comparison(axes[0], dem_file, stream_file, gdf_serial, 
                              "Serial Optimized", f"Time: {time_serial:.2f}s")
                
                plot_comparison(axes[1], dem_file, stream_file, gdf_parallel,
                              "Parallel Adaptive (Speed)", f"Time: {time_parallel:.2f}s | Speedup: {speedup:.2f}x")
                
                fig.suptitle(f"Performance Comparison - Subset {subset}\nIoU: {iou:.4f}", 
                            fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.05)
                
                viz_output = os.path.join(output_dir, f"comparison_serial_vs_parallel_{subset}.png")
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
    parser = argparse.ArgumentParser(description="Compare Serial vs Parallel Performance")
    parser.add_argument("--subsets", nargs="+", type=int, default=[1, 5, 10, 15, 25, 50, 100], 
                        help="List of subsets to process (default: 1 5 10 15 25 50 100)")
    parser.add_argument("--dem-file", default="data/dem_byneset_5m.tif", help="Path to DEM file")
    parser.add_argument("--output-dir", default="output/comparison_serial_optimized_vs_parallel_adaptive_speed", 
                        help="Directory to save results")
    args = parser.parse_args()
    
    run_comparison(args.subsets, args.dem_file, args.output_dir)


if __name__ == "__main__":
    main()
