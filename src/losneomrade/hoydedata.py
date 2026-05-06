"""Høydedata DEM fetching module.

Functions for downloading Digital Elevation Models (DEM) from the
Høydedata.no ImageServer REST API.
"""

import logging
import time
from urllib.request import HTTPError, urlopen

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import MemoryFile
from shapely.geometry.base import BaseGeometry

from .config import HoydedataConfig, settings

logger = logging.getLogger(__name__)


def get_hoydedata(
    bounds: tuple,
    layer: str | None = None,
    res: int | None = None,
    nodata: int | None = None,
    max_retries: int | None = None,
    config: HoydedataConfig | None = None,
) -> dict:
    """Download DEM from www.høydedata.no.

    Args:
        bounds: Bounding box as (xmin, xmax, ymin, ymax).
        layer: Høydedata API layer name. Defaults to config value.
        res: Resolution of the output DEM in meters. Defaults to config value.
        nodata: Value for nodata pixels. Defaults to config value.
        max_retries: Maximum retry attempts. Defaults to config value.
        config: Optional HoydedataConfig override. Uses global settings if None.

    Returns:
        Dictionary with keys: windows_dem_arrays, windows_transforms, windows,
        profile, full_array.

    Raises:
        Exception: If DEM cannot be fetched after max retries.
    """
    cfg = config or settings.hoydedata
    layer = layer or cfg.layer
    res = res if res is not None else cfg.resolution
    nodata = nodata if nodata is not None else cfg.nodata
    max_retries = max_retries if max_retries is not None else cfg.max_retries

    if layer not in settings.valid_layers:
        logger.error(f"Invalid API layer: {layer}")
        return {}

    xmin, xmax, ymin, ymax = bounds
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)

    request_url = (
        f"{cfg.base_url}/{layer}/ImageServer/exportImage?"
        f"bbox={xmin},{ymin},{xmax},{ymax}&size={width},{height}&bboxSR=&size=&imageSR=&time=&format=tiff&"
        f"pixelType=F32&noData={nodata}&noDataInterpretation=esriNoDataMatchAny"
        f"&interpolation=+RSP_BilinearInterpolation&compression=&compressionQuality=&"
        f"bandIds=&mosaicRule=&renderingRule=&f=image"
    )

    attempts = 0
    while attempts < max_retries:
        try:
            tif_bytes = urlopen(request_url).read()
            break
        except Exception:
            attempts += 1
            time.sleep(cfg.retry_wait)
    else:
        logger.error(f"Failed to fetch DEM after {max_retries} attempts. URL: {request_url}")
        raise Exception("Error (Probably area requested is too big/small or høydedata is down)")

    windows_dems = []
    windows_transforms = []

    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as dataset:
                dataset_profile = dataset.profile
                windows = [window for ij, window in dataset.block_windows()]
                full_array = dataset.read(1)
                for window in windows:
                    windows_dems.append(dataset.read(1, window=window))
                    windows_transforms.append(rasterio.windows.transform(window, dataset.transform))
    except Exception:
        logger.error(f"Error reading DEM response. URL: {request_url}")
        raise

    return {
        "windows_dem_arrays": windows_dems,
        "windows_transforms": windows_transforms,
        "windows": windows,
        "profile": dataset_profile,
        "full_array": full_array,
    }


def profile(
    line: BaseGeometry,
    dtm_layer: str | None = None,
    nodata: int | None = None,
    fra_crs: int = 4326,
    to_crs: int = 25833,
    config: HoydedataConfig | None = None,
) -> np.ndarray:
    """Compute a terrain profile from a given line by fetching DEM data.

    Args:
        line: Shapely geometry with xy coordinates defining the profile line.
        dtm_layer: Høydedata API layer. Defaults to config value.
        nodata: Value for nodata pixels. Defaults to config value.
        fra_crs: Input CRS EPSG code.
        to_crs: Output CRS EPSG code.
        config: Optional HoydedataConfig override. Uses global settings if None.

    Returns:
        Numpy array with columns [X, Y, Z, cumulative_distance].
    """
    cfg = config or settings.hoydedata
    dtm_layer = dtm_layer or cfg.layer
    nodata = nodata if nodata is not None else cfg.nodata

    retries = 10
    wait = 5

    gdf = gpd.GeoDataFrame(index=[0], crs=f"epsg:{int(fra_crs)}", geometry=[line]).to_crs(epsg=to_crs)
    linea2 = gdf.iloc[0].geometry
    n_points = int(max(linea2.length / 5, 5))
    new_points = [linea2.interpolate(i / float(n_points - 1), normalized=True) for i in range(n_points)]
    points_coords = np.array([[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points])

    xmin, ymin = np.min(points_coords, axis=0) - 10
    xmax, ymax = np.max(points_coords, axis=0) + 10
    width = int((xmax - xmin) / 5)
    height = int((ymax - ymin) / 5)

    request_url = (
        f"{cfg.base_url}/{dtm_layer}/ImageServer/exportImage?"
        f"bbox={xmin},{ymin},{xmax},{ymax}&size={width},"
        f"{height}&bboxSR=&size=&imageSR=&time=&format=tiff&pixelType=F32&"
        f"noData={nodata}&noDataInterpretation=esriNoDataMatchAny"
        f"&interpolation=+RSP_BilinearInterpolation&compression=&"
        f"compressionQuality=&bandIds=&mosaicRule=&renderingRule=&f=image"
    )

    attempt = 0
    while attempt < retries:
        try:
            tif_bytes = urlopen(request_url).read()
            break
        except HTTPError:
            attempt += 1
            logger.debug(f"Attempt {attempt} failed, retrying...")
            time.sleep(wait)
    if attempt == retries:
        raise Exception("HTTPError")

    z_dem = []
    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as dataset:
                dem_array = dataset.read(1)
                for pp in points_coords:
                    ind = dataset.index(pp[0], pp[1])
                    z_dem.append(dem_array[ind[0], ind[1]])
    except rasterio.errors.RasterioIOError:
        logger.error("Failed to read DEM raster for profile extraction")

    cum_dist = np.cumsum(np.sqrt(np.sum((np.r_[[[0, 0]], np.diff(points_coords, axis=0)[:, :2]]) ** 2, axis=1)))

    points = np.c_[points_coords, z_dem, cum_dist]
    return points
