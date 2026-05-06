# Configuration

The package uses a `Settings` dataclass that provides sensible defaults
but can be customized.

## Default settings

```python
from losneomrade.config import settings

# Høydedata (DEM fetching)
settings.hoydedata.base_url   # "https://hoydedata.no/arcgis/rest/services"
settings.hoydedata.layer      # "NHM_DTM_25833"
settings.hoydedata.resolution # 5 (meters)
settings.hoydedata.nodata     # -9999
settings.hoydedata.max_retries  # 5
settings.hoydedata.retry_wait   # 1 (second)

# MSML masks
settings.msml.base_url   # "https://gis4.nve.no/map/rest/services/Mapservices/MSML/MapServer"
settings.msml.layer_id   # 0
settings.msml.max_records # 2000
```

## Overriding settings

You can modify settings at runtime:

```python
from losneomrade.config import settings

# Use a different DEM layer
settings.hoydedata.layer = "dtm1_33_wcs"
settings.hoydedata.resolution = 1

# Point to a different MSML endpoint
settings.msml.base_url = "https://my-internal-server/MSML/MapServer"
```

## Using custom rasters

All main functions accept a `custom_raster` parameter that bypasses
Høydedata entirely:

```python
result = terrain_criteria.run_terrain_criteria(
    bounds=None,  # not needed with custom_raster
    source=source,
    custom_raster="path/to/my_dem.tif",
)
```

This is the recommended approach for:

- Offline/reproducible analysis
- Working with DEMs you already have
- Testing

## CRS

Everything is in **EPSG:25833** (UTM zone 33N). The DEM, source geometries,
masks, and results all use this CRS. Make sure your inputs are projected
to 25833 before passing them in.
