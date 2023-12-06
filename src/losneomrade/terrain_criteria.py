import bisect
import warnings
import tempfile
from typing import Union

import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import shapes

from . import utils

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_terrain_criteria(bounds: tuple,
                         source: Union[gpd.GeoDataFrame, np.ndarray],
                         source_depth: float = 0.0,
                         clip_to_msml: bool = False,
                         h_min: float = 5,
                         custom_raster=None) -> gpd.GeoDataFrame:
    """
    Wrapper function for running terrain criteria calculations
    Args:
        bounds: xmin,xmax,ymin,ymax of the calculation window. None if custom_raster is used.
        source: geodataframe with the source points (LineStrings or Points), 
                or a numpy array with the coordinates of the source points.
        source_depth: depth of the source points
        clip_to_msml: wheter to clip against MSML (sammenhengede forekomster).
        h_min: minumun height for calculations, default 5m
        custom_raster: path to a custom raster file (tif) to use for calculations

    Returns: geodataframe with the polygonized terrain criteria results
    """
    if isinstance(source, gpd.GeoDataFrame):
        if np.all(source.geom_type == "LineString"):
            source_points = generate_source_points(source)
        elif np.all(source.geom_type == "Point"):
            source_points = source.get_coordinates().values
    elif isinstance(source, np.ndarray):
        source_points = source

    with tempfile.TemporaryDirectory() as tempdir:
        tc = terrain_criteria(
            bounds=bounds,
            points=source_points,
            point_depth=source_depth,
            out_filename=tempdir+'/tc',
            clip_to_msml=clip_to_msml,
            h_min=h_min,
            custom_raster=custom_raster
        )

    return tc


def generate_source_points(polylines: gpd.GeoDataFrame, distance_chainage: float = 5) -> np.ndarray:
    """
    Generate source points from geodataframe of LineStrings
    Args:
        polylines: geodataframe of polylines (LineStrings)
        distance_chainage: distance between source points in meters

    Returns:
        points_coords: numpy array with the coordinates (x,y) of the source points
    """
    points_coords_list = []
    for pline in polylines.itertuples():

        line_gdf = pline.geometry
        length = line_gdf.length

        n_points = int(max(length / distance_chainage, 5))
        new_points = [line_gdf.interpolate(i / float(n_points - 1), normalized=True)
                      for i in range(n_points)
                      ]
        points_coords_list.append(
            [[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points]
        )

    points_coords = np.concatenate(points_coords_list)

    return points_coords


def terrain_criteria(bounds: tuple,
                     points: np.ndarray,
                     out_filename: str,
                     point_depth: float = 0.0,
                     clip_to_msml=False,
                     h_min: float = 5,
                     reclassify_results=True,
                     custom_raster=None) -> gpd.GeoDataFrame:
    """
    Run calculation of terrain criteria
    Args:
        bounds: xmin,xmax,ymin,ymax of the calculation window
        points: array with the source points (x,y,z)
        out_filename: path to save results (without extention)
        point_depth: depth of the source points
        clip_to_msml: wheter to clip against MSML (sammenhengede forekomster).

    Returns: Saves a tif and a GeoJson file with the results (areas that fills the terrain criteria), returns
             a geodataframe.

    """
    out_filename = out_filename.split(".")[0]  # keep the name without extension

    if custom_raster is None:
        try:
            window_data = utils.get_hoydedata(bounds)
        except MemoryError:
            print("Error: Maybe høydedata is down or your area is too big.")
            raise
    else:
        window_data = utils.generate_windows(custom_raster)

    windows = window_data["windows"]
    windows_dems = window_data["windows_dem_arrays"]
    windows_transforms = window_data["windows_transforms"]
    raster_profile = window_data["profile"]
    nan_value = raster_profile["nodata"]

    points = utils.set_z_from_raster(points, window_data)
    points[:, 2] = points[:, 2] - point_depth

    try:
        with rasterio.open(f"{out_filename}.tif", "w+", **raster_profile) as out:
            for index, window in enumerate(windows):
                results_window = compute_from_windows(windows_dems[index], windows_transforms[index], points,
                                                      nan_value, h_min, reclassify_results=reclassify_results)
                out.write(results_window, window=window, indexes=1)

            result_raster = out.read(1)
            raster_transform = out.transform
    except Exception as e:
        print("Error writing output raster.")
        raise

    gpd_polygonized_raster = polygonize_terrain_criteria(result_raster, raster_transform)

    if clip_to_msml:
        gpd_polygonized_raster = clip_results_to_msml(gpd_polygonized_raster, bounds)

    return gpd_polygonized_raster


def compute_from_windows(dem_data, transform, source_points, nan_value=-9999, h_min=5, reclassify_results=True):
    """
    Computes the terrain criteria for a given raster window
    Args:
        dem_data: array with elevations of the current window
        transform: transform associated with the current window
        source_points: x,y,z coordinates of the source points
        nan_value: raster's nodata value
        h_min: minumun height for calculations, default 5m
        reclassify_results: wheter to return a classified result or a continuous slope value

    Returns: numpy array with classified (or raw) slopes values

    """
    results_slope = np.ones_like(dem_data) * nan_value

    if np.all(dem_data == nan_value):
        return results_slope
    if source_points.size == 0:
        return results_slope  # skip blocks with no source points

    coords = utils.dem_coordinates(dem_data, transform)
    results_slope = utils.compute_slope(coords, source_points, h_min=h_min, nodata=nan_value).reshape(dem_data.shape)

    if reclassify_results:
        results_slope = reclass(results_slope)

    return results_slope


def reclass(matrix: np.ndarray) -> np.ndarray:
    """
    Reclassifies the terrain criteria results
    Args:
        matrix: numpy array with the terrain criteria results

    Returns: numpy array with the reclassified terrain criteria results
    """
    classes = [0.05, 0.067, 0.2, 0.33, 1.7, 1000]
    #         1:20   1:15   1:5   1:3  60 degrees
    reclass_vectorized = np.vectorize(lambda x: bisect.bisect_left(classes, x))

    return reclass_vectorized(matrix)


def clip_results_to_msml(results_gpd: gpd.GeoDataFrame, bounds: tuple) -> gpd.GeoDataFrame:
    """
    Clips results to NGU's MSML layer from NVE's MapServer
    Args:
        results_gpd: Geodataframe with the results from aktsomhet calculation
        bounds: bounds of the calculation area (xmin, xmax, ymin, ymax)

    Returns:
    Clipped results as geodataframe
    """
    xmin, xmax, ymin, ymax = bounds

    # concatenate masks
    mask = utils.get_msml_mask(bounds=(xmin, ymin, xmax, ymax))

    # clip results
    results_gpd = gpd.clip(results_gpd, mask)

    return results_gpd


def polygonize_terrain_criteria(
        result_raster: np.ndarray, raster_transform: rasterio.transform, crs: int = 25833
) -> gpd.GeoDataFrame:
    """
    Polygonizes the terrain criteria results, filters away flat areas (slope class < 1, slope<1:20)
    Args:
        result_raster: numpy array with the terrain criteria results
        raster_transform: transform of the raster
        crs: crs of the raster

    Returns: geodataframe with the polygonized terrain criteria results

    """

    results = ({"properties": {"slope": int(v)}, "geometry": s}
               for _, (s, v) in enumerate(shapes(result_raster.astype(np.int16), mask=None, transform=raster_transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.slope >= 1]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(epsg=crs)

    return gpd_polygonized_raster
