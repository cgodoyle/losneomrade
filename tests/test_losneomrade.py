import tempfile
import unittest

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString

from losneomrade import utils, terrain_criteria, retrogression


# runs from parent directory: python -m unittest

class TestLosneomrade(unittest.TestCase):
    def setUp(self) -> None:
        self.bound_alna = 268463.9, 270007.6, 6651396.2, 6652564.4
        self.points_alna = np.array([(268883.5400622159, 6651785.961672257),
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
                                     (269573.36843977927, 6651894.129818862)])

    def test_basic_slope(self):
        print("\ntest basic slope with terrain criteria")
        with tempfile.TemporaryDirectory() as tempdir:
            dem, profile = utils.generate_fake_slope(100, 100, 2000, 150, 1/5, 2e5, 6e6)
            with rasterio.open(tempdir + "/fake_slope.tif", "w", **profile) as src:
                src.write(dem, 1)

            tc = terrain_criteria.terrain_criteria(
                bounds=None,
                points=np.array([[2e5+100, 6e6-1150, 100]]),
                point_depth=0.5,
                out_filename=tempdir+'/tc',
                clip_to_msml=False,
                h_min=0,
                custom_raster=tempdir + "/fake_slope.tif",

            )

            template = gpd.read_file("tests/testdata/template_1_5.geojson")
            self.assertTrue(tc[["geometry", "slope"]].equals(template[["geometry", "slope"]]))

    def test_basic_slope_retrogression(self):
        print("\ntest basic slope with retrogression")
        with tempfile.TemporaryDirectory() as tempdir:
            dem, profile = utils.generate_fake_slope(100, 100, 2000, 150, 1/5, 2e5, 6e6)
            with rasterio.open(tempdir + "/fake_slope.tif", "w", **profile) as src:
                src.write(dem, 1)
            rel = retrogression.run_retrogression(bounds=None,
                                                  rel_shape=gpd.GeoDataFrame(geometry=[Point(2e5+100, 6e6-1150, 100)]),
                                                  point_depth=0.5,
                                                  clip_to_msml=False,
                                                  custom_raster=tempdir + "/fake_slope.tif",
                                                  min_slope=1/5,
                                                  min_length=75,
                                                  min_height=0,
                                                  )
        template = gpd.read_file("tests/testdata/template_retro_1_5.geojson")
        self.assertTrue(rel[["geometry"]].equals(template[["geometry"]]))

    def test_slope_hoydedata(self):
        print("\ntest slope from høydedata - Alna")
        xmin, xmax, ymin, ymax = self.bound_alna
        points = self.points_alna
        with tempfile.TemporaryDirectory() as tempdir:
            tc = terrain_criteria.terrain_criteria(
                bounds=(xmin, xmax, ymin, ymax),
                points=points,
                point_depth=0.5,
                out_filename=tempdir+'/tc',
                clip_to_msml=True,
                h_min=0,

            )

        self.assertIsNotNone(tc)
        self.assertGreaterEqual(tc.area.sum(), 700_000)
        self.assertGreaterEqual(tc[tc.slope == 1].area.sum(), 200_000)
        self.assertGreaterEqual(tc[tc.slope == 2].area.sum(), 400_000)
        self.assertGreaterEqual(tc[tc.slope == 3].area.sum(), 60_000)
        self.assertGreaterEqual(tc[tc.slope == 4].area.sum(), 9_000)
    #   self.assertAlmostEquals(tc[tc.slope == 5].area.sum(), 0)

    def test_slope_hoydedata_retro(self):
        print("\ntest slope from høydedata retrogression - Alna")
        xmin, xmax, ymin, ymax = self.bound_alna
        line_coords = self.points_alna
        rel_lines = source_line = gpd.GeoDataFrame(geometry=[LineString(line_coords)], crs=25833)

        rel = retrogression.run_retrogression(
            bounds=(xmin, xmax, ymin, ymax),
            rel_shape=rel_lines,
            point_depth=0.5,
            clip_to_msml=True,
            min_slope=1/15,
            min_length=75,
            min_height=0,
            return_animation=False,
        )
        self.assertIsNotNone(rel)
        self.assertGreaterEqual(rel.area.sum(), 500_000)

        rel = retrogression.run_retrogression(
            bounds=(xmin, xmax, ymin, ymax),
            rel_shape=rel_lines,
            point_depth=0.5,
            clip_to_msml=True,
            min_slope=1/5,
            min_length=75,
            min_height=0,
            return_animation=False,
        )
        self.assertIsNotNone(rel)
        self.assertGreaterEqual(rel.area.sum(), 100_000)
