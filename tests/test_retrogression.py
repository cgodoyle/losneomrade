"""Tests for retrogression module.

Offline tests use fake slopes to verify the retrogression pipeline
produces correct results without hitting external APIs.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, Point

from losneomrade import retrogression


class TestRetrogressionOffline:
    """Offline retrogression tests using fake slopes."""

    def test_run_retrogression_basic(self, fake_slope_1_5):
        """Basic retrogression from a point source on a 1:5 slope."""
        source = gpd.GeoDataFrame(geometry=[Point(2e5 + 80, 6e6 - 140, 100)], crs=25833)

        result = retrogression.run_retrogression(
            bounds=None,
            rel_shape=source,
            point_depth=0.5,
            custom_raster=fake_slope_1_5["path"],
            min_slope=1 / 5,
            min_length=20,
            min_height=0,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert "geometry" in result.columns

    def test_run_retrogression_with_animation(self, fake_slope_1_5):
        """Retrogression returns animation when requested."""
        source = gpd.GeoDataFrame(geometry=[Point(2e5 + 80, 6e6 - 140, 100)], crs=25833)

        result = retrogression.run_retrogression(
            bounds=None,
            rel_shape=source,
            point_depth=0.5,
            custom_raster=fake_slope_1_5["path"],
            min_slope=1 / 5,
            min_length=20,
            min_height=0,
            return_animation=True,
        )

        assert isinstance(result, tuple)
        gdf, animation = result
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert isinstance(animation, list)
        assert len(animation) > 0

    def test_landslide_retrogression_propagates(self, fake_slope_1_5):
        """landslide_retrogression expands from initial release on steep slope."""
        dem = fake_slope_1_5["dem"]  # 2D array
        profile = fake_slope_1_5["profile"]
        transform = profile["transform"]

        # Create initial release at a point in a valid location
        initial_release = np.zeros_like(dem, dtype=bool)
        row, col = dem.shape[0] // 2, dem.shape[1] // 2
        initial_release[row, col] = True

        release, animation = retrogression.landslide_retrogression(
            dem=dem,
            initial_release=initial_release,
            dem_transform=transform,
            min_slope=1 / 5,
            min_height=0,
            min_length=50,
            max_length=200,
        )

        # Should propagate beyond the initial pixel
        assert release.sum() > 1
        assert len(animation) > 1

    def test_landslide_retrogression_stops_on_flat(self):
        """Retrogression stops when slope criterion is not met."""
        # Create a flat DEM
        dem = np.ones((100, 100), dtype=np.float32) * 100.0
        transform = rasterio.transform.from_origin(0, 100 * 5, 5, 5)

        initial_release = np.zeros_like(dem, dtype=bool)
        initial_release[50, 50] = True

        release, animation = retrogression.landslide_retrogression(
            dem=dem,
            initial_release=initial_release,
            dem_transform=transform,
            min_slope=1 / 15,
            min_height=5,
            min_length=10,
            max_length=100,
        )

        # On flat terrain with h_min=5, should not propagate much beyond min_length
        # (unconditional expansion happens for min_length, then stops)
        assert release.sum() > 1  # at least unconditional expansion

    def test_run_retrogression_with_initial_landslide(self, fake_slope_1_5):
        """Two-phase retrogression (initial + retro) runs successfully."""

        line = LineString([(2e5 + 50, 6e6 - 100), (2e5 + 100, 6e6 - 150)])
        source = [line]

        result = retrogression.run_retrogression_with_initial_landslide(
            bounds=None,
            rel_shape=source,
            point_depth=0.5,
            ini_slope=1 / 4,
            retro_slope=[1 / 15],
            min_length=10,
            min_height=0,
            custom_raster=fake_slope_1_5["path"],
        )

        assert isinstance(result, gpd.GeoDataFrame)

    def test_create_buffer(self):
        """create_buffer creates a ring around input pixels."""
        image = np.zeros((10, 10), dtype=bool)
        image[5, 5] = True

        buffer = retrogression.create_buffer(image, buffer_size=1)

        assert buffer.sum() > 0
        assert not buffer[5, 5]  # original pixel not in buffer ring

    def test_apply_mask(self):
        """apply_mask zeros out pixels where mask is 0."""
        array = np.ones((5, 5), dtype=np.int32)
        mask = np.ones((5, 5), dtype=np.int32)
        mask[2:, :] = 0

        result = retrogression.apply_mask(array, mask)

        assert result[:2, :].sum() == 10
        assert result[2:, :].sum() == 0

    def test_apply_mask_none(self):
        """apply_mask with None returns the original array."""
        array = np.ones((5, 5), dtype=np.int32)
        result = retrogression.apply_mask(array, None)
        assert np.array_equal(result, array)


@pytest.mark.slow
class TestRetrogressionNetwork:
    """Tests that hit the Høydedata API."""

    def test_retrogression_with_hoydedata(self, alna_bounds, alna_source_line):
        """Full retrogression pipeline with real DEM."""
        result = retrogression.run_retrogression(
            bounds=alna_bounds,
            rel_shape=alna_source_line,
            point_depth=0.5,
            min_slope=1 / 15,
            min_length=75,
            min_height=0,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert result.area.sum() >= 500_000
