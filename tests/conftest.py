"""Shared fixtures for losneomrade tests."""

import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString

from losneomrade import utils


@pytest.fixture
def fake_slope_1_5():
    """Generate a fake slope with 1:5 gradient and write to a temp raster.

    Returns a dict with dem array, profile, and temp raster path.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        dem, profile = utils.generate_fake_slope(30, 100, 100, 130, 1 / 5, 2e5, 6e6)
        raster_path = f"{tempdir}/fake_slope_1_5.tif"
        with rasterio.open(raster_path, "w", **profile) as src:
            src.write(dem, 1)
        yield {"dem": dem, "profile": profile, "path": raster_path}


@pytest.fixture
def fake_slope_1_15():
    """Generate a fake slope with 1:15 gradient and write to a temp raster.

    Returns a dict with dem array, profile, and temp raster path.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        dem, profile = utils.generate_fake_slope(30, 100, 100, 130, 1 / 15, 2e5, 6e6)
        raster_path = f"{tempdir}/fake_slope_1_15.tif"
        with rasterio.open(raster_path, "w", **profile) as src:
            src.write(dem, 1)
        yield {"dem": dem, "profile": profile, "path": raster_path}


@pytest.fixture
def alna_bounds():
    """Bounds for Alna test area (xmin, xmax, ymin, ymax)."""
    return 268463.9, 270007.6, 6651396.2, 6652564.4


@pytest.fixture
def alna_source_line():
    """Source line for Alna test area as a GeoDataFrame."""
    line_coords = np.array(
        [
            (268883.5400622159, 6651785.961672257),
            (268917.3041989159, 6651614.686395486),
            (268952.34858209593, 6651598.08903631),
            (269098.9750438174, 6651692.086224091),
            (269136.0121169521, 6651688.541504248),
            (269187.3338556726, 6651662.554160272),
            (269266.9063958938, 6651685.158933432),
            (269327.31869611563, 6651689.869289508),
            (269385.2269397425, 6651692.168949791),
            (269430.1044651593, 6651718.1729600765),
            (269417.70676140685, 6651750.211168494),
            (269417.9726969357, 6651774.200699851),
            (269471.1996794914, 6651798.455341154),
            (269514.28554422176, 6651813.7414565245),
            (269552.6144975672, 6651849.758094928),
            (269566.83348302596, 6651865.730493411),
            (269573.36843977927, 6651894.129818862),
        ]
    )
    return gpd.GeoDataFrame(geometry=[LineString(line_coords)], crs=25833)
