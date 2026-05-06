"""Tests for terrain_criteria module.

These tests use fake slopes generated offline to verify that the terrain
criteria pipeline produces correct results without hitting external APIs.
"""

import tempfile

import geopandas as gpd
import numpy as np
import pytest

from losneomrade import terrain_criteria


class TestTerrainCriteriaOffline:
    """Offline tests using fake slopes."""

    def test_run_terrain_criteria_with_line_source(self, fake_slope_1_5):
        """Terrain criteria runs and returns GeoDataFrame when source is a LineString."""
        from shapely.geometry import LineString

        line = LineString([(2e5 + 50, 6e6 - 100), (2e5 + 100, 6e6 - 150)])
        source = gpd.GeoDataFrame(geometry=[line], crs=25833)

        result = terrain_criteria.run_terrain_criteria(
            bounds=None,
            source=source,
            source_depth=0.5,
            h_min=0,
            custom_raster=fake_slope_1_5["path"],
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert "geometry" in result.columns
        assert "slope" in result.columns
        assert len(result) > 0

    def test_run_terrain_criteria_with_numpy_source(self, fake_slope_1_5):
        """Terrain criteria runs with numpy array source points."""
        points = np.array([[2e5 + 80, 6e6 - 140]])

        result = terrain_criteria.run_terrain_criteria(
            bounds=None,
            source=points,
            source_depth=0.5,
            h_min=0,
            custom_raster=fake_slope_1_5["path"],
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_run_terrain_criteria_with_point_source(self, fake_slope_1_5):
        """Terrain criteria runs with Point geometry source."""
        from shapely.geometry import Point

        source = gpd.GeoDataFrame(geometry=[Point(2e5 + 80, 6e6 - 140)], crs=25833)

        result = terrain_criteria.run_terrain_criteria(
            bounds=None,
            source=source,
            source_depth=0.5,
            h_min=0,
            custom_raster=fake_slope_1_5["path"],
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_terrain_criteria_produces_expected_slope_classes(self, fake_slope_1_5):
        """Terrain criteria produces reasonable slope classes for a 1:5 slope."""
        points = np.array([[2e5 + 80, 6e6 - 140]])

        with tempfile.TemporaryDirectory() as tempdir:
            result = terrain_criteria.terrain_criteria(
                bounds=None,
                points=points,
                point_depth=0.5,
                out_filename=f"{tempdir}/tc",
                    h_min=0,
                custom_raster=fake_slope_1_5["path"],
            )

        assert isinstance(result, gpd.GeoDataFrame)
        # A 1:5 slope should produce slope classes >= 3 (1:5 = 0.2, class boundary)
        assert result["slope"].max() >= 3

    def test_terrain_criteria_reclass_disabled(self, fake_slope_1_5):
        """When reclassify_results=False, returns continuous slope values in the raster."""
        points = np.array([[2e5 + 80, 6e6 - 140]])

        with tempfile.TemporaryDirectory() as tempdir:
            result = terrain_criteria.terrain_criteria(
                bounds=None,
                points=points,
                point_depth=0.5,
                out_filename=f"{tempdir}/tc",
                    h_min=0,
                reclassify_results=False,
                custom_raster=fake_slope_1_5["path"],
            )

        # With continuous values, result may be empty because polygonize filters < 1
        # The important thing is it doesn't crash
        assert isinstance(result, gpd.GeoDataFrame)

    def test_generate_source_points(self):
        """generate_source_points produces correct output from a LineString."""
        from shapely.geometry import LineString

        line = LineString([(0, 0), (100, 0)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs=25833)

        points = terrain_criteria.generate_source_points(gdf, distance_chainage=10)

        assert isinstance(points, np.ndarray)
        assert points.shape[1] == 2
        assert len(points) >= 5  # minimum enforced in function

    def test_reclass_default_classes(self):
        """reclass produces expected class values."""
        # 0.04 → class 0 (below 1:20)
        # 0.06 → class 1 (between 1:20 and 1:15)
        # 0.1 → class 2 (between 1:15 and 1:5)
        # 0.25 → class 3 (between 1:5 and 1:3)
        matrix = np.array([[0.04, 0.06], [0.1, 0.25]])
        result = terrain_criteria.reclass(matrix)

        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 2
        assert result[1, 1] == 3


@pytest.mark.slow
class TestTerrainCriteriaNetwork:
    """Tests that hit the Høydedata API (slow, require network)."""

    def test_terrain_criteria_with_hoydedata(self, alna_bounds, alna_source_line):
        """Full terrain criteria pipeline with real DEM from Høydedata."""
        result = terrain_criteria.run_terrain_criteria(
            bounds=alna_bounds,
            source=alna_source_line,
            source_depth=0.5,
            h_min=0,
        )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert result.area.sum() >= 700_000
