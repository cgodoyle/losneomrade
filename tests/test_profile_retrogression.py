"""Tests for profile_retrogression module.

Uses fake slopes to verify the profile-based retrogression pipeline.
"""

import tempfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString

from losneomrade import utils
from losneomrade.profile_retrogression import retrogression_from_profiles


class TestProfileRetrogressionOffline:
    """Offline tests for profile retrogression."""

    @pytest.fixture
    def steep_slope_data(self):
        """Generate a steep slope (1:5) suitable for profile retrogression testing."""
        with tempfile.TemporaryDirectory() as tempdir:
            dem, profile = utils.generate_fake_slope(200, 200, 3000, 200, 1 / 5, 2e5, 6e6)
            raster_path = f"{tempdir}/steep.tif"
            with rasterio.open(raster_path, "w", **profile) as src:
                src.write(dem, 1)
            yield {"dem": dem, "profile": profile, "path": raster_path}

    def test_retrogression_from_profiles_basic(self, steep_slope_data):
        """Profile retrogression runs and returns a GeoDataFrame."""
        # Create a line roughly in the middle of the slope
        line = LineString([(2e5 + 200, 6e6 - 1500), (2e5 + 800, 6e6 - 1500)])

        result, debug = retrogression_from_profiles(
            line=line,
            dem_array=steep_slope_data["dem"],
            dem_profile=steep_slope_data["profile"],
            slope_ratio=1 / 15,
            n_profiles=5,
            profile_length_m=300,
            side="right",
            depth_m=1,
            tolerance_distance_m=50,
            min_height_m=5,
            max_height_m=30,
            debug=True,
        )

        assert isinstance(result, gpd.GeoDataFrame)

    def test_retrogression_from_profiles_both_sides(self, steep_slope_data):
        """Profile retrogression works for both 'left' and 'right' sides."""
        line = LineString([(2e5 + 200, 6e6 - 1500), (2e5 + 800, 6e6 - 1500)])

        for side in ["left", "right"]:
            result, _ = retrogression_from_profiles(
                line=line,
                dem_array=steep_slope_data["dem"],
                dem_profile=steep_slope_data["profile"],
                slope_ratio=1 / 15,
                n_profiles=3,
                profile_length_m=200,
                side=side,
                depth_m=1,
                tolerance_distance_m=50,
                min_height_m=5,
                max_height_m=30,
                debug=True,
            )

            assert isinstance(result, gpd.GeoDataFrame)

    def test_retrogression_from_profiles_returns_debug_layers(self, steep_slope_data):
        """Debug mode returns profile layers when retrogression succeeds."""
        line = LineString([(2e5 + 200, 6e6 - 1500), (2e5 + 800, 6e6 - 1500)])

        result, debug = retrogression_from_profiles(
            line=line,
            dem_array=steep_slope_data["dem"],
            dem_profile=steep_slope_data["profile"],
            slope_ratio=1 / 15,
            n_profiles=5,
            profile_length_m=300,
            side="right",
            depth_m=1,
            tolerance_distance_m=50,
            min_height_m=5,
            max_height_m=30,
            debug=True,
        )

        # Debug is None when no valid retrogression points found, or a tuple of 4 layers
        assert isinstance(result, gpd.GeoDataFrame)
        if debug is not None:
            assert len(debug) == 4

    def test_retrogression_from_profiles_empty_on_flat(self):
        """On a flat DEM, profile retrogression should return empty or minimal result."""
        # Create a flat DEM
        dem = np.ones((100, 100), dtype=np.float32) * 100.0
        transform = rasterio.transform.from_origin(0, 500, 5, 5)
        profile = {
            "driver": "GTiff",
            "height": 100,
            "width": 100,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:25833",
            "transform": transform,
            "nodata": -9999,
        }

        line = LineString([(100, 250), (400, 250)])

        result, _ = retrogression_from_profiles(
            line=line,
            dem_array=dem,
            dem_profile=profile,
            slope_ratio=1 / 15,
            n_profiles=5,
            profile_length_m=100,
            side="right",
            depth_m=1,
            tolerance_distance_m=50,
            min_height_m=5,
            max_height_m=30,
            debug=True,
        )

        # On flat terrain, no retrogression should occur
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0 or result.area.sum() < 1
