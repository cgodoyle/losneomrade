"""Tests for core utility functions.

Tests the functions most likely to break from code changes:
slope computation, rasterize/polygonize, config, and fake slope generation.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from losneomrade import utils
from losneomrade.config import HoydedataConfig, settings
from losneomrade.hoydedata import get_hoydedata


class TestConfig:
    """Tests for the configuration system."""

    def test_default_settings(self):
        """Default settings have expected values."""
        assert settings.hoydedata.layer == "NHM_DTM_25833"
        assert settings.hoydedata.resolution == 5
        assert settings.hoydedata.nodata == -9999
        assert "hoydedata.no" in settings.hoydedata.base_url

    def test_custom_config(self):
        """Custom config overrides defaults."""
        custom = HoydedataConfig(layer="dtm1_33_wcs", resolution=1)
        assert custom.layer == "dtm1_33_wcs"
        assert custom.resolution == 1

    def test_valid_layers(self):
        """Valid layers list contains expected entries."""
        assert "NHM_DTM_25833" in settings.valid_layers
        assert "NHM_DTM_25832" in settings.valid_layers


class TestGenerateFakeSlope:
    """Tests for generate_fake_slope utility."""

    def test_generates_valid_raster(self):
        """generate_fake_slope produces a valid DEM array and profile."""
        dem, profile = utils.generate_fake_slope(50, 50, 1000, 100, 1 / 10, 1e5, 5e6)

        assert isinstance(dem, np.ndarray)
        assert dem.ndim == 2
        assert dem.shape[0] > 0
        assert dem.shape[1] > 0
        assert isinstance(profile, dict)
        assert "transform" in profile
        assert "crs" in profile

    def test_slope_gradient(self):
        """Generated slope has a non-constant elevation (not flat)."""
        dem, profile = utils.generate_fake_slope(100, 100, 2000, 150, 1 / 5, 2e5, 6e6)

        # The slope should not be flat — there should be elevation variation
        assert dem.max() > dem.min()


class TestSlopeComputation:
    """Tests for compute_slope and compute_slope_chunked."""

    def test_compute_slope_basic(self):
        """compute_slope produces valid slope values."""
        # 3 DEM pixels: (0,0,100), (10,0,99), (20,0,98) — slope ~1:10
        coords = np.array([[0, 0, 100], [10, 0, 99], [20, 0, 98]], dtype=np.float64)
        # Source point at origin, lower elevation
        points = np.array([[0, 0, 90]], dtype=np.float64)

        slopes = utils.compute_slope(coords, points, h_min=0)

        assert isinstance(slopes, np.ndarray)
        assert len(slopes) == 3

    def test_compute_slope_chunked_matches_regular(self):
        """compute_slope_chunked produces same results as compute_slope."""
        np.random.seed(42)
        coords = np.column_stack(
            [
                np.random.uniform(0, 100, 50),
                np.random.uniform(0, 100, 50),
                np.random.uniform(50, 150, 50),
            ]
        )
        points = np.array([[50, 50, 40], [30, 30, 35]], dtype=np.float64)

        regular = utils.compute_slope(coords, points, h_min=0)
        chunked = utils.compute_slope_chunked(coords, points, h_min=0, chunk_size=10)

        np.testing.assert_array_almost_equal(regular, chunked)


class TestRasterizePolygonize:
    """Tests for rasterize_shape and polygonize_results."""

    def test_rasterize_geodataframe(self):
        """rasterize_shape produces binary array from GeoDataFrame."""
        geom = box(10, 10, 40, 40)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=25833)
        profile = {
            "height": 100,
            "width": 100,
            "transform": rasterio.transform.from_origin(0, 500, 5, 5),
        }

        result = utils.rasterize_shape(gdf, profile)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)
        assert result.max() == 1
        assert result.sum() > 0

    def test_rasterize_geometry_list(self):
        """rasterize_shape works with a list of geometries."""
        geom = box(10, 10, 40, 40)
        profile = {
            "height": 100,
            "width": 100,
            "transform": rasterio.transform.from_origin(0, 500, 5, 5),
        }

        result = utils.rasterize_shape([geom], profile)

        assert result.sum() > 0

    def test_polygonize_results(self):
        """polygonize_results converts binary array to GeoDataFrame."""
        array = np.zeros((10, 10), dtype=np.int16)
        array[3:7, 3:7] = 1  # a square block
        profile = {
            "transform": rasterio.transform.from_origin(0, 50, 5, 5),
            "crs": "EPSG:25833",
        }

        result = utils.polygonize_results(array, profile)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0


class TestGenerateWindows:
    """Tests for generate_windows (custom raster loading)."""

    def test_generate_windows_returns_expected_keys(self, fake_slope_1_5):
        """generate_windows returns dict with expected structure."""
        result = utils.generate_windows(fake_slope_1_5["path"])

        assert isinstance(result, dict)
        assert "windows_dem_arrays" in result
        assert "windows_transforms" in result
        assert "windows" in result
        assert "profile" in result
        assert "full_array" in result


class TestDemCoordinates:
    """Tests for dem_coordinates utility."""

    def test_dem_coordinates_shape(self):
        """dem_coordinates returns correct shape."""
        dem = np.ones((10, 10), dtype=np.float32) * 100
        transform = rasterio.transform.from_origin(0, 50, 5, 5)

        coords = utils.dem_coordinates(dem, transform)

        assert coords.shape == (100, 3)  # 10*10 pixels, 3 columns (x, y, z)


@pytest.mark.slow
class TestHoydedataNetwork:
    """Tests that hit the Høydedata API."""

    def test_get_hoydedata_returns_expected_structure(self):
        """get_hoydedata returns dict with expected keys."""
        bounds = (268463.9, 270007.6, 6651396.2, 6652564.4)
        result = get_hoydedata(bounds)

        assert isinstance(result, dict)
        assert "windows_dem_arrays" in result
        assert "windows_transforms" in result
        assert "windows" in result
        assert "profile" in result
        assert "full_array" in result
        assert isinstance(result["full_array"], np.ndarray)

    def test_get_hoydedata_invalid_layer(self):
        """get_hoydedata returns empty dict for invalid layer."""
        bounds = (268463.9, 270007.6, 6651396.2, 6652564.4)
        result = get_hoydedata(bounds, layer="invalid_layer")
        assert result == {}


@pytest.mark.slow
class TestMSMLNetwork:
    """Tests for MSML/maringrense fetching."""

    def test_get_msml_mask(self):
        """get_msml_mask returns a GeoDataFrame."""
        bounds = (267867.5, 6645040.8, 311812.6, 6677133.8)
        result = utils.get_msml_mask(bounds)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
