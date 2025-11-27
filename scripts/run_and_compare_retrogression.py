"""
Efficient Retrogression Analysis and Comparison Script

This script:
1. Checks for existing serial baseline results (skips if present)
2. Runs adaptive parallel methods
3. Compares results (timing, accuracy, visualization)
4. Generates comprehensive summary with plots

Usage:
    python scripts/run_and_compare_retrogression.py --stream-file data/streams_subset_5.geojson
    python scripts/run_and_compare_retrogression.py --stream-file data/streams_subset_5.geojson --force-baseline
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

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from losneomrade import terrain_criteria, retrogression
except ImportError:
    print("Could not import losneomrade. Make sure the 'src' directory is correctly structured.")
    sys.exit(1)


def plot_comparison(ax, dem_file, stream_file, retro_file, title, subtitle=""):
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
    if os.path.exists(retro_file):
        retro_gdf = gpd.read_file(retro_file)
        if not retro_gdf.empty:
            retro_gdf.plot(ax=ax, color='red', alpha=0.6, edgecolor='darkred', zorder=4)
            
            n_polygons = len(retro_gdf)
            area_ha = retro_gdf.geometry.area.sum() / 10000
            subtitle = f"{n_polygons} polygons, {area_ha:.2f} ha"
        else:
            subtitle = "No results"
    else:
        subtitle = "File not found"

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


def main():
    parser = argparse.ArgumentParser(description="Efficient Retrogression Analysis and Comparison")
    parser.add_argument("--stream-file", required=True, help="Path to stream input file")
    parser.add_argument("--dem-file", default="data/dem_byneset_5m.tif", help="Path to DEM file")
    parser.add_argument("--output-dir", default="output", help="Directory to save results")
    parser.add_argument("--force-baseline", action="store_true", 
                        help="Force re-run of baseline even if output exists")
    parser.add_argument("--skip-visualization", action="store_true",
                        help="Skip generating visualization plots")
    parser.add_argument("--adaptive-mode", default="balanced", 
                        choices=["speed", "balanced", "accuracy", "serial"],
                        help="Adaptive strategy priority (default: balanced)")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("EFFICIENT RETROGRESSION ANALYSIS AND COMPARISON")
    print("="*80)
    
    if args.adaptive_mode != "serial":
        print("\n[NOTE] Parallel methods are now optimized and accurate.")
        print("       This script will compare results to confirm accuracy.")
        print("       For guaranteed accuracy, use: --adaptive-mode serial")

    # Configuration
    STREAM_FILE = args.stream_file
    stream_filename = Path(STREAM_FILE).stem
    CUSTOM_DEM = args.dem_file

    # Parameters
    SOURCE_DEPTH = 0.0
    MIN_HEIGHT = 5.0
    MIN_SLOPE = 1 / 15
    MIN_LENGTH = 75.0
    CLIP_TO_MSML = False
    BUFFER_DISTANCE = 300
    POINTS_PER_METER = 1 / 10
    INITIAL_BUFFER = 10

    print(f"\nConfiguration:")
    print(f"  - Stream file: {STREAM_FILE}")
    print(f"  - DEM file: {CUSTOM_DEM}")
    print(f"  - Adaptive mode: {args.adaptive_mode}")

    # Validate inputs
    if not os.path.exists(STREAM_FILE):
        print(f"\n[ERROR] Stream file not found: {STREAM_FILE}")
        sys.exit(1)

    if not os.path.exists(CUSTOM_DEM):
        print(f"\n[ERROR] DEM file not found: {CUSTOM_DEM}")
        sys.exit(1)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Define output files
    baseline_output = os.path.join(args.output_dir, f"retrogression_serial_{stream_filename}.geojson")
    adaptive_output = os.path.join(args.output_dir, f"retrogression_adaptive_{args.adaptive_mode}_{stream_filename}.geojson")
    
    # Load streams
    print(f"\nLoading streams...")
    streams = gpd.read_file(STREAM_FILE)
    print(f"[OK] Loaded {len(streams)} stream features")

    # Calculate bounds
    bounds_array = streams.total_bounds
    bounds = (
        bounds_array[0] - BUFFER_DISTANCE,
        bounds_array[2] + BUFFER_DISTANCE,
        bounds_array[1] - BUFFER_DISTANCE,
        bounds_array[3] + BUFFER_DISTANCE,
    )

    # Generate source points
    print("\nGenerating source points...")
    try:
        source_points = terrain_criteria.generate_source_points(
            streams,
            distance_chainage=1 / POINTS_PER_METER,
        )
        print(f"[OK] Generated {len(source_points):,} source points")
    except Exception as e:
        print(f"\n[ERROR] generating source points: {e}")
        sys.exit(1)

    # Create initial release zones
    print("\nCreating initial release zones...")
    try:
        point_geometries = [Point(x, y) for x, y in source_points[:, :2]]
        points_gdf = gpd.GeoDataFrame(geometry=point_geometries, crs=streams.crs)
        buffered = points_gdf.buffer(INITIAL_BUFFER)
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered, crs=streams.crs)
        initial_release = buffered_gdf.dissolve()
        print(f"[OK] Created initial release zones")
    except Exception as e:
        print(f"\n[ERROR] creating initial release zones: {e}")
        sys.exit(1)

    # Common kwargs
    base_kwargs = {
        "bounds": bounds,
        "rel_shape": initial_release,
        "point_depth": SOURCE_DEPTH,
        "clip_to_msml": CLIP_TO_MSML,
        "min_slope": MIN_SLOPE,
        "min_height": MIN_HEIGHT,
        "min_length": MIN_LENGTH,
        "custom_raster": CUSTOM_DEM,
    }

    results = {}
    timings = {}

    # ========================================================================
    # 1. BASELINE (SERIAL) - Check if exists first
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE (SERIAL)")
    print("="*80)
    
    if os.path.exists(baseline_output) and not args.force_baseline:
        print(f"\n[SKIP] Baseline result already exists: {baseline_output}")
        print("       Loading existing result... (use --force-baseline to re-run)")
        try:
            baseline_result = gpd.read_file(baseline_output)
            timings["Serial (Baseline)"] = None  # Unknown time
            results["Serial (Baseline)"] = baseline_result
            
            area_ha = baseline_result.geometry.area.sum() / 10000
            print(f"[OK] Loaded baseline result")
            print(f"     - Polygons: {len(baseline_result)}")
            print(f"     - Area: {area_ha:.2f} ha")
        except Exception as e:
            print(f"[ERROR] Failed to load existing baseline: {e}")
            print("        Will re-run baseline...")
            args.force_baseline = True

    if not os.path.exists(baseline_output) or args.force_baseline:
        print("\nRunning serial baseline analysis...")
        start = time.time()
        try:
            baseline_result = retrogression.run_retrogression(
                **base_kwargs,
                return_animation=False,
                verbose=False
            )
            elapsed = time.time() - start
            timings["Serial (Baseline)"] = elapsed
            
            area_ha = baseline_result.geometry.area.sum() / 10000
            print(f"\n[OK] Completed in {elapsed:.2f}s")
            print(f"     - Polygons: {len(baseline_result)}")
            print(f"     - Area: {area_ha:.2f} ha")
            
            baseline_result.to_file(baseline_output, driver="GeoJSON")
            print(f"     - Saved to: {baseline_output}")
            
            results["Serial (Baseline)"] = baseline_result
        except Exception as e:
            print(f"\n[ERROR] Serial baseline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # ========================================================================
    # 2. ADAPTIVE PARALLEL (skip if serial mode)
    # ========================================================================
    if args.adaptive_mode != "serial":
        print("\n" + "="*80)
        print(f"ADAPTIVE PARALLEL ({args.adaptive_mode.upper()} MODE)")
        print("="*80)
        
        print(f"\nRunning adaptive parallel analysis...")
        start = time.time()
        try:
            adaptive_result = retrogression.run_retrogression_parallel_adaptive(
                **base_kwargs,
                speed_priority=args.adaptive_mode
            )
            elapsed = time.time() - start
            timings[f"Adaptive ({args.adaptive_mode})"] = elapsed
            
            area_ha = adaptive_result.geometry.area.sum() / 10000
            print(f"\n[OK] Completed in {elapsed:.2f}s")
            print(f"     - Polygons: {len(adaptive_result)}")
            print(f"     - Area: {area_ha:.2f} ha")
            
            adaptive_result.to_file(adaptive_output, driver="GeoJSON")
            print(f"     - Saved to: {adaptive_output}")
            
            results[f"Adaptive ({args.adaptive_mode})"] = adaptive_result
        except Exception as e:
            print(f"\n[ERROR] Adaptive parallel failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*80)
        print("SERIAL MODE - SKIPPING PARALLEL COMPARISON")
        print("="*80)
        print("\nUsing serial method only (most accurate approach).")
        print("To test parallel methods, use: --adaptive-mode balanced or accuracy")

    # ========================================================================
    # 3. COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Performance comparison
    if timings.get("Serial (Baseline)") is not None:
        baseline_time = timings["Serial (Baseline)"]
        adaptive_time = timings.get(f"Adaptive ({args.adaptive_mode})")
        
        if adaptive_time:
            speedup = baseline_time / adaptive_time
            print(f"\nPerformance:")
            print(f"  Serial (Baseline):     {baseline_time:>8.2f}s")
            print(f"  Adaptive ({args.adaptive_mode}):  {adaptive_time:>8.2f}s  ({speedup:.2f}x speedup)")
    else:
        print("\nPerformance:")
        print(f"  Serial (Baseline):     (loaded from cache)")
        adaptive_time = timings.get(f"Adaptive ({args.adaptive_mode})")
        if adaptive_time:
            print(f"  Adaptive ({args.adaptive_mode}):  {adaptive_time:>8.2f}s")

    # Accuracy comparison
    if "Serial (Baseline)" in results and f"Adaptive ({args.adaptive_mode})" in results:
        baseline_area = results["Serial (Baseline)"].geometry.area.sum()
        adaptive_area = results[f"Adaptive ({args.adaptive_mode})"].geometry.area.sum()
        diff_pct = ((adaptive_area - baseline_area) / baseline_area * 100) if baseline_area > 0 else 0
        
        print(f"\nAccuracy:")
        print(f"  Serial area:     {baseline_area/10000:>8.2f} ha")
        print(f"  Adaptive area:   {adaptive_area/10000:>8.2f} ha  ({diff_pct:+.2f}%)")
        
        if abs(diff_pct) < 1:
            print(f"  Status:          [OK] Excellent - differences < 1%")
        elif abs(diff_pct) < 10:
            print(f"  Status:          [OK] Good - differences < 10%")
        elif abs(diff_pct) < 50:
            print(f"  Status:          [WARN] Warning - significant differences")
        else:
            print(f"  Status:          [POOR] Poor - large differences (components interact strongly)")
            print(f"\n  Recommendation: Use Serial method for this dataset")

    # ========================================================================
    # 4. VISUALIZATION
    # ========================================================================
    if not args.skip_visualization:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATION")
        print("="*80)
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            
            print("\nPlotting serial baseline...")
            plot_comparison(axes[0], CUSTOM_DEM, STREAM_FILE, baseline_output, 
                          "Serial (Baseline)")
            
            print("Plotting adaptive parallel...")
            plot_comparison(axes[1], CUSTOM_DEM, STREAM_FILE, adaptive_output,
                          f"Adaptive Parallel ({args.adaptive_mode})")
            
            # Overall title with summary
            summary = ""
            if timings.get("Serial (Baseline)") is not None and f"Adaptive ({args.adaptive_mode})" in timings:
                speedup = timings["Serial (Baseline)"] / timings[f"Adaptive ({args.adaptive_mode})"]
                summary = f"Speedup: {speedup:.2f}x"
            
            if "Serial (Baseline)" in results and f"Adaptive ({args.adaptive_mode})" in results:
                baseline_area = results["Serial (Baseline)"].geometry.area.sum()
                adaptive_area = results[f"Adaptive ({args.adaptive_mode})"].geometry.area.sum()
                diff_pct = ((adaptive_area - baseline_area) / baseline_area * 100) if baseline_area > 0 else 0
                if summary:
                    summary += f" | "
                summary += f"Area diff: {diff_pct:+.1f}%"
            
            fig.suptitle(f"Retrogression Analysis Comparison\n{summary}", 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save
            viz_output = os.path.join(args.output_dir, 
                                     f"comparison_serial_vs_adaptive_{args.adaptive_mode}_{stream_filename}.png")
            plt.savefig(viz_output, dpi=300, bbox_inches='tight')
            print(f"\n[OK] Visualization saved to: {viz_output}")
            
            plt.close()
            
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # 5. FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - Baseline:      {baseline_output}")
    print(f"  - Adaptive:      {adaptive_output}")
    if not args.skip_visualization:
        viz_output = os.path.join(args.output_dir, 
                                 f"comparison_serial_vs_adaptive_{args.adaptive_mode}_{stream_filename}.png")
        print(f"  - Visualization: {viz_output}")
    
    print("\nRecommendations:")
    if "Serial (Baseline)" in results and f"Adaptive ({args.adaptive_mode})" in results:
        baseline_area = results["Serial (Baseline)"].geometry.area.sum()
        adaptive_area = results[f"Adaptive ({args.adaptive_mode})"].geometry.area.sum()
        diff_pct = abs((adaptive_area - baseline_area) / baseline_area * 100) if baseline_area > 0 else 0
        
        if diff_pct < 5:
            print(f"  [OK] Excellent! Adaptive method works well for this dataset")
            print(f"       Difference: {diff_pct:.1f}% (acceptable)")
            print(f"       >> Use adaptive mode for {args.adaptive_mode} speedup with good accuracy")
        elif diff_pct < 15:
            print(f"  [CAUTION] Moderate differences detected ({diff_pct:.1f}%)")
            print(f"            This may be acceptable for exploratory work")
            print(f"            >> Use serial method for production/accurate results")
            if args.adaptive_mode == "speed":
                print(f"            >> Or try: --adaptive-mode accuracy (more conservative grouping)")
        else:
            print(f"  [FAIL] Large differences detected ({diff_pct:.1f}%)")
            print(f"         Components STRONGLY INTERACT - parallel splitting causes major errors")
            print(f"         >> DO NOT USE adaptive methods for this dataset")
            print(f"         >> ALWAYS USE serial method: --adaptive-mode serial")
            print(f"")
            print(f"         Why this happens: Retrogression from different sources overlaps")
            print(f"         and interacts. Parallel methods process them separately, missing")
            print(f"         these interactions and significantly underpredicting the affected area.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
