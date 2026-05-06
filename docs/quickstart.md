# Quick Start

## Installation

```bash
pip install git+https://github.com/cgodoyle/losneomrade.git
```

For development (clone and install in editable mode):

```bash
git clone https://github.com/cgodoyle/losneomrade.git
cd losneomrade
pip install -e ".[dev]"
```

## Breaking changes

If you are updating older scripts, the main changes to watch for are:

- `bounds` now always use `(xmin, ymin, xmax, ymax)`
- `clip_to_msml` was removed in favor of `mask=...`
- mask helpers now live under `losneomrade.masks`

## Basic usage

All bounds use standard GIS order `(xmin, ymin, xmax, ymax)` in EPSG:25833.

### Terrain Criteria (pixel-based)

Slope is computed for each pixel in the DEM relative to the source points.
Results are reclassified into slope interval classes.

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from losneomrade import terrain_criteria, masks, utils

# Define bounds and source line (EPSG:25833)
bounds = (268463.9, 6651396.2, 270007.6, 6652564.4)

source_line = gpd.GeoDataFrame(
    geometry=[LineString([
        (268883.5, 6651785.9),
        (268917.3, 6651614.7),
        (269098.9, 6651692.1),
        (269266.9, 6651685.2),
        (269430.1, 6651718.2),
        (269573.4, 6651894.1),
    ])],
    crs=25833,
)

# Fetch MSML mask from NVE
mask = masks.get_msml_mask(bounds)

# Run terrain criteria
tc = terrain_criteria.run_terrain_criteria(
    bounds=bounds,
    source=source_line,
    source_depth=0.5,
    mask=mask,
    h_min=5,
)

# tc is a GeoDataFrame with classified slope polygons
print(tc.head())
```

You can also pass source as a numpy array of coordinates or a GeoDataFrame of Points.

### Retrogression (step-wise propagation)

Propagates a release area outward following slope criteria (1:15 by default):

```python
from losneomrade import retrogression

retro, animation = retrogression.run_retrogression(
    bounds=bounds,
    rel_shape=source_line,
    point_depth=0.5,
    mask=mask,
    min_slope=1 / 15,
    min_height=5,
    min_length=75,
    return_animation=True,
)

print(f"Release area: {retro.area.sum():.0f} m²")
```

### Retrogression with initial landslide (erosion-triggered)

For erosion-triggered landslides, use a steeper initial slope (e.g. 1:4) to model
the first failure, then continue with the full retrogressive slope (1:15):

```python
retro, animation = retrogression.run_retrogression_with_initial_landslide(
    bounds=bounds,
    rel_shape=source_line,
    point_depth=0.5,
    mask=mask,
    ini_slope=1 / 4,
    retro_slope=[1 / 15],
    min_height=0,
    min_length=20,
    return_animation=True,
)
```

This two-phase approach models a scenario where an initial steep erosion event
triggers subsequent retrogressive failures at the gentler 1:15 slope.

### Using a custom raster

All methods accept `custom_raster` to bypass the Høydedata API:

```python
tc = terrain_criteria.run_terrain_criteria(
    bounds=None,
    source=source_line,
    custom_raster="path/to/dem.tif",
)
```

### Profile Retrogression

Analyze retrogression along cross-section profiles:

```python
from losneomrade.profile_retrogression import retrogression_from_profiles

result, debug = retrogression_from_profiles(
    profiles=profiles_gdf,
    dem_path="path/to/dem.tif",
    mask_array=mask_raster,
    source_line=source_line,
)
```
