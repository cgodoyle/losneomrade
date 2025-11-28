"""
Generate Benchmark Data Visualization
"""
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
import rasterio
import geopandas as gpd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath("src"))

def plot_benchmark_data(dem_file, stream_file, output_file):
    print(f"Generating benchmark plot...")
    print(f"DEM: {dem_file}")
    print(f"Streams: {stream_file}")
    
    # Load DEM
    with rasterio.open(dem_file) as src:
        dem_data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
    # Load Streams
    streams = gpd.read_file(stream_file)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_data, vert_exag=1, dx=5, dy=5)
    ax.imshow(hs, cmap='gray', extent=extent, origin='upper', alpha=1.0)
    
    # Streams
    streams.plot(ax=ax, color='blue', linewidth=1.0, alpha=0.7, label='Streams')
    
    # Formatting
    ax.set_title("Benchmark Dataset: Byneset", fontsize=16, fontweight='bold')
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
    # Legend
    legend_elements = [Line2D([0], [0], color='blue', lw=1.5, label='Stream Network')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Scale bar (approximate)
    # ax.text(0.05, 0.05, "Scale: 1km", transform=ax.transAxes, fontsize=12, 
    #         bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    dem_file = "data/dem_byneset_5m.tif"
    stream_file = "data/streams_as_source.geojson"
    output_file = "docs/images/benchmark_data_overview.png"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    plot_benchmark_data(dem_file, stream_file, output_file)
