"""Mask fetching and manipulation for losneomrade.

Provides functions to retrieve MSML (mulig sammenhengende marin leire),
AR5 land-use masks, and utilities for combining/modifying masks.
"""

import logging

import geopandas as gpd
import requests
from shapely.geometry import box

from losneomrade.config import settings

logger = logging.getLogger(__name__)


def get_msml_mask(
    bounds: tuple[float, float, float, float],
    max_records: int = 2000,
) -> gpd.GeoDataFrame:
    """Fetch MSML polygons from NVE MapServer within the given bounds.

    Uses the new NVE MSML MapServer (2025 recipe) which serves data
    directly in EPSG:25833.

    Args:
        bounds: Bounding box as ``(xmin, ymin, xmax, ymax)`` in EPSG:25833.
        max_records: Maximum records per request page.

    Returns:
        GeoDataFrame with dissolved MSML polygons, or empty GeoDataFrame
        if no features are found.
    """
    xmin, ymin, xmax, ymax = bounds
    url = f"{settings.msml.base_url}/{settings.msml.layer_id}/query"

    params = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25833",
        "outSR": "25833",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OBJECTID",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": max_records,
    }

    features: list = []
    result_offset = 0

    while True:
        params["resultOffset"] = result_offset
        logger.debug("Fetching MSML: offset=%d, bounds=(%.0f,%.0f,%.0f,%.0f)", result_offset, xmin, ymin, xmax, ymax)

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        batch = data.get("features", [])
        features.extend(batch)

        if data.get("exceededTransferLimit") or len(batch) == max_records:
            result_offset += len(batch)
        else:
            break

    if not features:
        logger.info("No MSML features found within bounds")
        return gpd.GeoDataFrame(geometry=[], crs=25833)

    gdf = gpd.GeoDataFrame.from_features(features, crs=25833)
    gdf = gpd.clip(gdf, box(xmin, ymin, xmax, ymax))
    return gdf.dissolve()


def get_ar5_mask(
    bounds: tuple[float, float, float, float],
    max_records: int = 2000,
) -> gpd.GeoDataFrame:
    """Fetch AR5 land-use features (rock/shallow soil) within bounds.

    Retrieves grunnforhold codes 42 (fjell i dagen) and 43 (grunnlendt)
    from NVE's FKB MapServer. This is an optional utility — not part of
    the main MSML masking workflow.

    Args:
        bounds: Bounding box as ``(xmin, ymin, xmax, ymax)`` in EPSG:25833.
        max_records: Maximum records per request page.

    Returns:
        GeoDataFrame with AR5 rock/shallow soil polygons.
    """
    xmin, ymin, xmax, ymax = bounds
    url = "https://gis3.nve.no/map/rest/services/Mapservices/FKB/MapServer/2/query"

    params = {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25833",
        "outSR": "25833",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "grunnforhold",
        "where": "grunnforhold IN (42, 43)",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": max_records,
    }

    features: list = []
    result_offset = 0

    while True:
        params["resultOffset"] = result_offset
        logger.debug("Fetching AR5: offset=%d", result_offset)

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        batch = data.get("features", [])
        features.extend(batch)

        if data.get("exceededTransferLimit") or len(batch) == max_records:
            result_offset += len(batch)
        else:
            break

    if not features:
        logger.info("No AR5 features found within bounds")
        return gpd.GeoDataFrame(geometry=[], crs=25833)

    gdf = gpd.GeoDataFrame.from_features(features, crs=25833)
    return gpd.clip(gdf, box(xmin, ymin, xmax, ymax))


def get_clipping_mask(
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame | None:
    """Get the MSML clipping mask for the given bounds.

    Convenience wrapper around ``get_msml_mask``.

    Args:
        bounds: Bounding box as ``(xmin, ymin, xmax, ymax)`` in EPSG:25833.

    Returns:
        MSML clipping mask GeoDataFrame, or ``None`` if no features found.
    """
    mask = get_msml_mask(bounds)
    if mask.empty:
        return None
    return mask


def modify_release_mask(
    release_mask: gpd.GeoDataFrame,
    no_release_mask: gpd.GeoDataFrame | None = None,
    sup_release_mask: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Modify a release mask with exclusion and supplementary areas.

    Args:
        release_mask: Base release mask.
        no_release_mask: Areas to subtract from the release mask.
        sup_release_mask: Areas to add to the release mask.

    Returns:
        Modified release mask.
    """
    if no_release_mask is not None:
        release_mask = gpd.GeoDataFrame(
            geometry=release_mask.dissolve().difference(no_release_mask.dissolve()),
            crs=release_mask.crs,
        )

    if sup_release_mask is not None:
        release_mask = gpd.GeoDataFrame(
            geometry=release_mask.dissolve().union(sup_release_mask.dissolve()),
            crs=release_mask.crs,
        )

    return release_mask


def check_msml_service() -> bool:
    """Check whether the MSML MapServer is reachable.

    Returns:
        ``True`` if the service responds successfully, ``False`` otherwise.
    """
    url = f"{settings.msml.base_url}/{settings.msml.layer_id}/query"
    params = {
        "geometry": "265122,6648110,266151,6648761",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25833",
        "outSR": "25833",
        "returnGeometry": "false",
        "returnCountOnly": "true",
        "f": "json",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return "count" in data and data.get("error") is None
    except (requests.RequestException, ValueError):
        return False
