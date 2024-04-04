import unittest

import geopandas as gpd

from losneomrade import utils

class TestUtils(unittest.TestCase):
    def setUp(self) -> None:

        self.bounds_balsfjord = 642004.6820,7707111.2311,663977.2379, 7723157.7108
        self.bounds_romerike = 267867.5038,6645040.8152,311812.6155,6677133.7745
        self.bounds_vaaler =  328621.7002,6721444.3011,339607.9781,6729467.5409


    def test_get_msml_balsfjord(self):
        """
        Test an area with both MSML and Area under MG, to check that it is fecthing all the data
        """
        print("\ntest getting Aktsomhet Marin Leire from Balsfjord\n")
        
        mm1 = gpd.clip(utils.get_maringrense(self.bounds_balsfjord, "msml"), self.bounds_balsfjord).dissolve()
        mm2 = gpd.clip(utils.get_maringrense(self.bounds_balsfjord, "area_under_mg"), self.bounds_balsfjord).dissolve()
        
        msml_mask = utils.get_msml_mask(self.bounds_balsfjord)

        area_msml = mm1.area.values.flatten().sum()
        area_umg = mm2.area.values.flatten().sum() 
        area_mask = msml_mask.area.values.flatten().sum()
               

        print(f"\tMSML area = {area_msml}")
        print(f"\tArea under MG = {area_umg}")
        print("\tAreas per april 2024 are estimated to be 18_000_000 and 13_000_000")
        print("\tthis can change and should be updated in the test")


        self.assertGreaterEqual(area_msml, 18_000_000)
        self.assertGreaterEqual(area_umg, 13_000_000)
        self.assertAlmostEqual(area_mask//1e6, area_msml//1e6+area_umg//1e6)

    def test_romerike(self):
        """
        Test a big area with only MSML. This checks that the function fetchs the data despite possible limits of the API
        """
        print("\ntest getting Aktsomhet Marin Leire from Romerike\n")
        
        mm1 = gpd.clip(utils.get_maringrense(self.bounds_romerike, "msml"), self.bounds_romerike).dissolve()
        mm2 = gpd.clip(utils.get_maringrense(self.bounds_romerike, "area_under_mg"), self.bounds_romerike).dissolve()
        msml_mask = utils.get_msml_mask(self.bounds_romerike)

        area_msml = mm1.area.values.flatten().sum()
        area_umg = mm2.area.values.flatten().sum()
        area_mask = msml_mask.area.values.flatten().sum()
               

        print(f"\tMSML area = {area_msml}")
        print("\tArea MSML per april 2024 is estimated to be 700_000_000")
        print("\tthis can change and should be updated in the test")


        
        self.assertGreaterEqual(area_msml, 700_000_000)
        self.assertAlmostEqual(area_umg, 0)
        self.assertAlmostEqual(area_mask//1e6, area_msml//1e6)

    
    def test_vaaler(self):
        """
        Test in an area with only Area under MG (Våler).
        """
        print("\ntest getting Aktsomhet Marin Leire from Våler\n")

        mm1 = gpd.clip(utils.get_maringrense(self.bounds_vaaler, "msml"), self.bounds_vaaler).dissolve()
        mm2 = gpd.clip(utils.get_maringrense(self.bounds_vaaler, "area_under_mg"), self.bounds_vaaler).dissolve()
        msml_mask = utils.get_msml_mask(self.bounds_vaaler)

        area_msml = mm1.area.values.flatten().sum()
        area_umg = mm2.area.values.flatten().sum()
        area_mask = msml_mask.area.values.flatten().sum()


        print(f"\tMSML area = {area_msml}")
        print("\tArea under MG per april 2024 is estimated to be 65_000_000")
        print("\tthis can change if mapping is done and will make the test fail. Update the test with the new value.")

        self.assertAlmostEqual(area_msml, 0)
        self.assertGreaterEqual(area_umg, 65_000_000)
        self.assertAlmostEqual(area_mask//1e6, area_umg//1e6)

