"""Type aliases for the losneomrade package."""

import os

import geopandas as gpd
import pandas as pd

PathLike = os.PathLike | str
DataFrameLike = pd.DataFrame | gpd.GeoDataFrame
