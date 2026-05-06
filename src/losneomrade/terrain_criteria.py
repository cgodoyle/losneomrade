import bisect
import logging
import tempfile
import warnings

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes

from . import utils

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def run_terrain_criteria(
    bounds: tuple | None,
    source: gpd.GeoDataFrame | np.ndarray,
    source_depth: float = 0.0,
    mask: gpd.GeoDataFrame | None = None,
    h_min: float = 5,
    reclassify_results: bool = True,
    classes: list[float] | None = None,
    custom_raster: str | None = None,
) -> gpd.GeoDataFrame:
    """Run terrain criteria calculations.

    Args:
        bounds: Bounding box as (xmin, xmax, ymin, ymax). None if custom_raster is used.
        source: GeoDataFrame with source points (LineStrings or Points),
            or a numpy array with source point coordinates.
        source_depth: Depth of the source points in meters.
        mask: Optional clipping mask as GeoDataFrame (e.g. from masks.get_msml_mask).
        h_min: Minimum height difference for calculations in meters.
        reclassify_results: Whether to return classified result or continuous slope value.
        classes: Slope values defining class boundaries for reclassification.
        custom_raster: Path to a custom raster file (tif) for calculations.

    Returns:
        GeoDataFrame with the polygonized terrain criteria results.
    """
    if isinstance(source, gpd.GeoDataFrame):
        if np.all(source.geom_type.isin(["LineString", "MultiLineString"])):
            source_points = generate_source_points(source)
        elif np.all(source.geom_type == "Point"):
            source_points = source.get_coordinates().values
        else:
            raise ValueError(
                "source must be a geodataframe of LineStrings/MultiLineStrings or Points"
            )  # I left this so we can track possible errors
    elif isinstance(source, np.ndarray):
        source_points = source

    with tempfile.TemporaryDirectory() as tempdir:
        tc = terrain_criteria(
            bounds=bounds,
            points=source_points,
            point_depth=source_depth,
            out_filename=tempdir + "/tc",
            mask=mask,
            h_min=h_min,
            reclassify_results=reclassify_results,
            classes=classes,
            custom_raster=custom_raster,
        )

    return tc


def generate_source_points(polylines: gpd.GeoDataFrame, distance_chainage: float = 5) -> np.ndarray:
    """Generate source points from a GeoDataFrame of LineStrings.

    Args:
        polylines: GeoDataFrame of polylines (LineStrings).
        distance_chainage: Distance between source points in meters.

    Returns:
        Numpy array with the coordinates (x, y) of the source points.
    """
    points_coords_list = []
    for _, geom in polylines[["geometry"]].iterrows():
        line_geom = geom.geometry
        length = line_geom.length

        n_points = int(max(length / distance_chainage, 5))
        new_points = [line_geom.interpolate(i / float(n_points - 1), normalized=True) for i in range(n_points)]
        points_coords_list.append(
            [[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points],
        )

    points_coords = np.concatenate(points_coords_list)

    return points_coords


def terrain_criteria(
    bounds: tuple | None,
    points: np.ndarray,
    out_filename: str,
    point_depth: float = 0.0,
    mask: gpd.GeoDataFrame | None = None,
    h_min: float = 5,
    reclassify_results: bool = True,
    classes: list[float] | None = None,
    custom_raster: str | None = None,
) -> gpd.GeoDataFrame:
    """Run terrain criteria calculation on a DEM.

    Args:
        bounds: Bounding box as (xmin, xmax, ymin, ymax).
        points: Array with the source points (x, y, z).
        out_filename: Path to save results (without extension).
        point_depth: Depth of the source points in meters.
        mask: Optional clipping mask as GeoDataFrame (e.g. from masks.get_msml_mask).
        h_min: Minimum height difference for calculations in meters.
        reclassify_results: Whether to return classified result or continuous slope value.
        classes: Slope values defining class boundaries for reclassification.
        custom_raster: Path to a custom raster file (tif) for calculations.

    Returns:
        GeoDataFrame with the polygonized terrain criteria results.
    """
    out_filename = out_filename.split(".")[0]  # keep the name without extension

    if custom_raster is None:
        assert bounds is not None, "bounds required when custom_raster is not provided"
        try:
            window_data = utils.get_hoydedata(bounds)
        except MemoryError:
            logger.error("Failed to fetch DEM. Høydedata may be down or area is too large.")
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
                results_window = compute_from_windows(
                    windows_dems[index],
                    windows_transforms[index],
                    points,
                    nan_value,
                    h_min,
                    reclassify_results=reclassify_results,
                    classes=classes,
                )
                out.write(results_window, window=window, indexes=1)

            result_raster = out.read(1)
            raster_transform = out.transform
    except Exception:
        logger.error("Error writing output raster.")
        raise

    gpd_polygonized_raster = polygonize_terrain_criteria(result_raster, raster_transform)

    if mask is not None:
        gpd_polygonized_raster = clip_results_to_mask(gpd_polygonized_raster, mask)

    return gpd_polygonized_raster


def compute_from_windows(
    dem_data: np.ndarray,
    transform: rasterio.transform.Affine,
    source_points: np.ndarray,
    nan_value: int = -9999,
    h_min: float = 5,
    reclassify_results: bool = True,
    classes: list[float] | None = None,
) -> np.ndarray:
    """Compute terrain criteria for a given raster window.

    Args:
        dem_data: Array with elevations of the current window.
        transform: Affine transform associated with the current window.
        source_points: Array with (x, y, z) coordinates of the source points.
        nan_value: Raster's nodata value.
        h_min: Minimum height difference for calculations in meters.
        reclassify_results: Whether to return classified result or continuous slope value.
        classes: Slope values defining class boundaries for reclassification.

    Returns:
        Numpy array with classified (or raw) slope values.
    """
    results_slope = np.ones_like(dem_data) * nan_value

    if np.all(dem_data == nan_value):
        return results_slope
    if source_points.size == 0:
        return results_slope  # skip blocks with no source points

    coords = utils.dem_coordinates(dem_data, transform)
    results_slope = utils.compute_slope(coords, source_points, h_min=h_min, nodata=nan_value).reshape(dem_data.shape)

    if reclassify_results:
        results_slope = reclass(results_slope, classes)

    return results_slope


def reclass(matrix: np.ndarray, classes: list[float] | None = None) -> np.ndarray:
    """Reclassify terrain criteria results into slope classes.

    Args:
        matrix: Numpy array with terrain criteria slope values.
        classes: Slope values defining class boundaries. Defaults to
            [0.05, 0.067, 0.2, 0.33, 1.7, 1000] (1:20, 1:15, 1:5, 1:3, 60°).

    Returns:
        Numpy array with reclassified terrain criteria results.
    """

    classes = [0.05, 0.067, 0.2, 0.33, 1.7, 1000] if classes is None else classes
    #         1:20   1:15   1:5   1:3  60 degrees
    reclass_vectorized = np.vectorize(lambda x: bisect.bisect_left(classes, x))

    return reclass_vectorized(matrix)


def clip_results_to_mask(results_gpd: gpd.GeoDataFrame, mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip terrain criteria results to a mask GeoDataFrame.

    Args:
        results_gpd: GeoDataFrame with the terrain criteria results.
        mask: Clipping mask as GeoDataFrame (e.g. from masks.get_msml_mask).

    Returns:
        Clipped results as GeoDataFrame.
    """
    return gpd.clip(results_gpd, mask)


def polygonize_terrain_criteria(
    result_raster: np.ndarray,
    raster_transform: rasterio.transform.Affine,
    crs: int = 25833,
) -> gpd.GeoDataFrame:
    """Polygonize terrain criteria results, filtering away flat areas.

    Removes areas with slope class < 1 (slope < 1:20).

    Args:
        result_raster: Numpy array with the terrain criteria results.
        raster_transform: Affine transform of the raster.
        crs: EPSG code of the raster CRS.

    Returns:
        GeoDataFrame with polygonized terrain criteria results.
    """

    results = (
        {"properties": {"slope": int(v)}, "geometry": s}
        for _, (s, v) in enumerate(shapes(result_raster.astype(np.int16), mask=None, transform=raster_transform))
    )
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.slope >= 1]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(epsg=crs)

    return gpd_polygonized_raster
