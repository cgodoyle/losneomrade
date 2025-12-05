import sys
import os
from pathlib import Path
#!/usr/bin/env python3
"""
Create a subset of streams for testing.
"""
import sys
import os
import argparse
from pathlib import Path
import geopandas as gpd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def main():
    parser = argparse.ArgumentParser(description='Create a subset of streams for testing')
    parser.add_argument('--input', default='data/streams_as_source.geojson', help='Input stream file')
    parser.add_argument('--count', type=int, default=3, help='Number of streams to include in subset')
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_file = os.path.join(base_dir, args.input)
    output_file = os.path.join(base_dir, f'data/streams_subset_{args.count}.geojson')
    
    print(f"Loading streams from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
        
    try:
        streams = gpd.read_file(input_file)
        
        print(f"Total streams: {len(streams)}")
        print(f"Selecting {args.count} random streams...")
        
        subset = streams.sample(n=args.count, random_state=42)
        
        subset.to_file(output_file, driver='GeoJSON')
        print(f"✓ Saved {len(subset)} streams to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
