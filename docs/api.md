# API Overview

## terrain_criteria

### `run_terrain_criteria()`

Main entry point for terrain criteria analysis.

```python
from losneomrade import terrain_criteria

result = terrain_criteria.run_terrain_criteria(
    bounds,            # (xmin, ymin, xmax, ymax) or None
    source,            # GeoDataFrame (Lines/Points) or numpy array
    source_depth=0.0,  # meters below surface
    mask=None,         # GeoDataFrame clipping mask
    h_min=5,           # min height difference (m)
    reclassify_results=True,
    classes=None,      # custom slope class boundaries
    custom_raster=None,  # path to local DEM .tif
)
# Returns: GeoDataFrame with 'slope' column (class values)
```

---

## retrogression

### `run_retrogression()`

Single-phase retrogression from a source area.

```python
from losneomrade import retrogression

result = retrogression.run_retrogression(
    bounds,             # (xmin, ymin, xmax, ymax) or None
    rel_shape,          # GeoDataFrame with source geometry
    point_depth=0.0,
    mask=None,          # GeoDataFrame clipping mask
    min_slope=1/15,
    min_height=5,
    min_length=75,      # minimum horizontal length of retrogression (m) to avoid stopping due to local flat terrain
    custom_raster=None,
    return_animation=False,
)
# Returns: GeoDataFrame (or tuple with animation frames)
```

### `run_retrogression_with_initial_landslide()`

Two-phase: steep initial failure + gentler retrogression.

```python
result = retrogression.run_retrogression_with_initial_landslide(
    bounds,
    rel_shape,          # list of shapely geometries
    point_depth=0.0,
    mask=None,
    ini_slope=1/4,      # steep initial phase
    retro_slope=1/15,   # or list of slopes
    min_height=5,
    min_length=75,
    custom_raster=None,
    return_animation=False,
)
```

---

## profile_retrogression

### `retrogression_from_profiles()`

Cross-section based retrogression analysis.

```python
from losneomrade.profile_retrogression import retrogression_from_profiles

result, debug_layers = retrogression_from_profiles(
    line,                         # source LineString / MultiLineString
    dem_array,                    # DEM as numpy array
    dem_profile,                  # raster profile/transform metadata
    slope_ratio=1/15,
    spacing_m=None,               # spacing between side profiles
    n_profiles=None,              # alternative to spacing_m
    profile_length_m=500,         # max profile length from source line
    side="right",                 # which side of the line to analyze
    depth_m=0,
    tolerance_distance_m=0,
    min_height_m=5,
    max_height_m=30,
    mask_array=None,              # 2D numpy array mask
    debug=False,
    min_n_points_for_envelope=3,
    clip_to_sourceline=True,
)
# Returns: (GeoDataFrame, tuple of debug layers or None)
```

Key parameters:

| Parameter | Meaning |
|-----------|---------|
| spacing_m | Target spacing between generated cross-sections along the source line. If omitted, the function derives it from `n_profiles`. |
| n_profiles | Optional number of cross-sections to generate along the source line. Used as an alternative to `spacing_m`. |
| side | Side of the source line where one-sided profiles are built, passed to the profile generator (`"right"` by default). |
| profile_length_m | Maximum length of each generated profile away from the source line. |
| depth_m | Initial vertical offset below the first terrain point before testing the retrogression slope line. |
| tolerance_distance_m | Distance window used when searching for the first valid retrogression line near the source before applying the full crossing test. This is to avoid stopping due to local flat terrain. (similar to `min_length` in `run_retrogression`) |
| min_height_m | Minimum relief that must develop along a profile for that profile to contribute to the final release envelope. |
| max_height_m | Upper bound on accepted relief along a profile. Profiles exceeding this height are ignored. Use `None` to disable the cap. |
| min_n_points_for_envelope | Minimum number of valid retrogression points required before building the final polygon envelope. |
| clip_to_sourceline | If `True`, trims generated profiles so they do not cross the source line before profiling terrain. |

---

## masks

### `get_msml_mask()`

Fetch MSML polygons from NVE MapServer.

```python
from losneomrade import masks

mask = masks.get_msml_mask(
    bounds,  # (xmin, ymin, xmax, ymax) in EPSG:25833
)
# Returns: GeoDataFrame (dissolved polygons, CRS 25833)
```

### `get_ar5_mask()`

Fetch AR5 rock/shallow soil areas (optional utility).

```python
ar5 = masks.get_ar5_mask(bounds)
```

### `check_msml_service()`

Check if the NVE MSML service is reachable.

```python
is_up = masks.check_msml_service()  # True/False
```

---

## hoydedata

### `get_hoydedata()`

Fetch DEM from Høydedata.no.

```python
from losneomrade.hoydedata import get_hoydedata

dem_data = get_hoydedata(
    bounds,       # (xmin, ymin, xmax, ymax)
    layer=None,   # defaults to settings.hoydedata.layer
)
# Returns: dict with 'full_array', 'profile', 'windows', etc.
```
