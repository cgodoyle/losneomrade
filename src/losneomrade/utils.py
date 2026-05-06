import logging
import os
import tempfile
import time
import warnings
from urllib.request import HTTPError, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio import MemoryFile
from rasterio.features import rasterize, shapes
from scipy.spatial import distance_matrix
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, split

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

HOYDEDATA_LAYER = "NHM_DTM_25833"


def get_hoydedata(
    bounds: tuple, layer: str = HOYDEDATA_LAYER, res: int = 5, nodata: int = -9999, max_retries=5
) -> dict:
    """
    Function for downloading DEM from www.høydedata.no.

    Args:
        bounds (tuple): Bounding box of the DEM to be downloaden in the form of xmin, xmax, ymin, ymax
        layer (str, optional): Which Høydedata API layer. "dtm1_32_wcs", "dtm1_33_wcs", "dtm1_32_wcs" or "dtm1_33_wcs".
                               Defaults to "dtm1_33_wcs".
        res (int, optional): Resolution of the output DEM in meters, if it is different from the layer used it
                            will be resampled. Defaults to 5.
        nodata (int, optional): Value for nodata pixels. Defaults to -9999.

    Returns:
        (tuple): (dem_array, transform): array with DEM values and the transform used to convert array's rows
                                         and cols to geographic coordinates.
    """

    # Check input layer
    if layer not in ["dtm1_32_wcs", "dtm1_33_wcs", "dtm10_32_wcs", "dtm10_33_wcs", "NHM_DTM_25833", "NHM_DTM_25832"]:
        print("Use a valid API layer (check help).")
        return dict()

    # Set up request to høydedata
    xmin, xmax, ymin, ymax = bounds
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)

    request_url = (
        f"https://hoydedata.no/arcgis/rest/services/{layer}/ImageServer/exportImage?"
        f"bbox={xmin},{ymin},{xmax},{ymax}&size={width},{height}&bboxSR=&size=&imageSR=&time=&format=tiff&"
        f"pixelType=F32&noData={nodata}&noDataInterpretation=esriNoDataMatchAny"
        f"&interpolation=+RSP_BilinearInterpolation&compression=&compressionQuality=&"
        f"bandIds=&mosaicRule=&renderingRule=&f=image"
    )

    # Open the request output with rasterio and save elevation array and transform
    attempts = 0
    wait_time = 1
    while attempts < max_retries:
        try:
            tif_bytes = urlopen(request_url).read()
            break
        except Exception:
            attempts += 1
            time.sleep(wait_time)
    else:
        print(request_url)
        raise Exception("Error (Probably area requested is too big/small or høydedata is down)")

    windows_dems = []
    windows_transforms = []

    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as dataset:
                dataset_profile = dataset.profile
                windows = [window for ij, window in dataset.block_windows()]
                full_array = dataset.read(1)
                for window in windows:
                    windows_dems.append(dataset.read(1, window=window))

                    windows_transforms.append(rasterio.windows.transform(window, dataset.transform))

    except Exception:
        print(request_url)
        print("Error (Probably area requested is too big/small or høydedata is down)")
        raise

    return {
        "windows_dem_arrays": windows_dems,
        "windows_transforms": windows_transforms,
        "windows": windows,
        "profile": dataset_profile,
        "full_array": full_array,
    }


def dem_coordinates(dem_array: np.ndarray, dem_transform: rasterio.transform.Affine) -> np.ndarray:
    """
    get coordinates of the given dem window
    Args:
        dem_array: window's elevation array
        dem_transform: window's transform

    Returns:
        coords: numpy array with the coordinates (x,y,z) of the dem
    """
    height, width = dem_array.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(dem_transform, rows, cols)
    x_dem = np.array(xs)
    y_dem = np.array(ys)
    coords = np.c_[x_dem.flatten(), y_dem.flatten(), dem_array.flatten()]
    return coords


def compute_slope(coords: np.ndarray, points: np.ndarray, h_min: float = 5, nodata: int = -9999) -> np.ndarray:
    """
    Compute the slopes of the given dem with respect to the (source) points
    Args:
        coords: dem window coordinates
        points: source point coordinates
        h_min: minimum height difference where slopes are calculated
        nodata: value given to pixels with no data

    Returns:
        max_slope: array with slopes (same shape as input dem)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy_1 = coords[:, :2]
        xy_2 = points[:, :2]
        z1 = coords[:, -1]
        z2 = points[:, -1]
        distance_mtx = distance_matrix(xy_1, xy_2)
        height_mtx = z1[:, np.newaxis] - z2
        hl_ratio = height_mtx / distance_mtx
        hl_ratio[height_mtx < h_min] = nodata
        max_slope = np.max(hl_ratio, axis=1)

        return max_slope


def compute_slope_chunked(
    coords: np.ndarray, points: np.ndarray, h_min: float = 5, nodata: int = -9999, chunk_size: int = 1000
) -> np.ndarray:
    """
    Compute the slopes of the given dem with respect to the (source) points using chunked processing
    to avoid memory explosion.
    Args:
        coords: dem window coordinates
        points: source point coordinates
        h_min: minimum height difference where slopes are calculated
        nodata: value given to pixels with no data
        chunk_size: number of points to process in each batch
    Returns:
        max_slope: array with slopes (same shape as input dem)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy_1 = coords[:, :2]
        z1 = coords[:, -1]

        max_slope = np.full(len(coords), nodata, dtype=np.float64)

        for i in range(0, len(points), chunk_size):
            chunk_points = points[i : i + chunk_size]
            xy_2 = chunk_points[:, :2]
            z2 = chunk_points[:, -1]

            distance_mtx = distance_matrix(xy_1, xy_2)
            height_mtx = z1[:, np.newaxis] - z2
            hl_ratio = height_mtx / distance_mtx
            hl_ratio[height_mtx < h_min] = nodata

            chunk_max = np.max(hl_ratio, axis=1)
            max_slope = np.maximum(max_slope, chunk_max)

        return max_slope


def set_z_from_raster(points_xy: np.ndarray, window_data: dict) -> np.ndarray:
    """
    Set elevation value to the given x,y points
    Args:
        points_xy: numpy array with the x,y coordinates to the points
        window_data: DEM-results from calling get_hoydedata function

    Returns: numpy array with x,y,z coordinates

    """
    height = window_data["profile"]["height"]
    width = window_data["profile"]["width"]
    dem_array = window_data["full_array"]
    transform = window_data["profile"]["transform"]

    bounds = rasterio.transform.array_bounds(height, width, transform)
    points_filt = points_xy[
        (points_xy[:, 0] > bounds[0])
        & (points_xy[:, 0] < bounds[2])
        & (points_xy[:, 1] > bounds[1])
        & (points_xy[:, 1] < bounds[3])
    ]

    index = np.array([rasterio.transform.rowcol(transform, xx[0], xx[1]) for xx in points_filt])
    z = np.array([dem_array[xx[0], xx[1]] for xx in index])
    filter_nan = z == window_data["profile"]["nodata"]

    return np.c_[points_filt[:, :2], z][~filter_nan]


def profile(line, dtm_layer=HOYDEDATA_LAYER, nodata=-9999, fra_crs=4326, to_crs=25833):
    """
    Compute a terrain profile from a given line
    Args:
        line: array with xy coordinates
        dtm_layer: høydedata api layer
        nodata: value to be used as nodata
        fra_crs: input crs
        to_crs: output crs

    Returns: numpy array with X, Y, Z, M values

    """
    retries = 10
    wait = 5

    attempt = 0

    gdf = gpd.GeoDataFrame(index=[0], crs=f"epsg:{int(fra_crs)}", geometry=[line]).to_crs(epsg=to_crs)
    linea2 = gdf.iloc[0].geometry
    n_points = int(max(linea2.length / 5, 5))
    new_points = [linea2.interpolate(i / float(n_points - 1), normalized=True) for i in range(n_points)]
    points_coords = np.array([[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points])

    xmin, ymin = np.min(points_coords, axis=0) - 10
    xmax, ymax = np.max(points_coords, axis=0) + 10
    width = int((xmax - xmin) / 5)
    height = int((ymax - ymin) / 5)

    request_url = (
        f"https://hoydedata.no/arcgis/rest/services/{dtm_layer}/ImageServer/exportImage?"
        f"bbox={xmin},{ymin},{xmax},{ymax}&size={width},"
        f"{height}&bboxSR=&size=&imageSR=&time=&format=tiff&pixelType=F32&"
        f"noData={nodata}&noDataInterpretation=esriNoDataMatchAny"
        f"&interpolation=+RSP_BilinearInterpolation&compression=&"
        f"compressionQuality=&bandIds=&mosaicRule=&renderingRule=&f=image"
    )
    while attempt < retries:
        try:
            tif_bytes = urlopen(request_url).read()
            break
        except HTTPError:
            attempt += 1
            # print(f"Attempt {attempt} failed. Error: {e}")
            time.sleep(wait)
    if attempt == retries:
        raise Exception("HTTPError")
    z_dem = []

    try:
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as dataset:
                dem_array = dataset.read(1)
                for pp in points_coords:
                    ind = dataset.index(pp[0], pp[1])
                    z_dem.append(dem_array[ind[0], ind[1]])

    except rasterio.errors.RasterioIOError:
        print("feil")

    cum_dist = np.cumsum(np.sqrt(np.sum((np.r_[[[0, 0]], np.diff(points_coords, axis=0)[:, :2]]) ** 2, axis=1)))

    points = np.c_[points_coords, z_dem, cum_dist]
    return points


def generate_plotly_profile(prof, max_depth=None, kp_depth=0, limit=15):
    """
    Generate a plotly figure with the profile
    Args:
        prof: numpy array with X, Y, Z, M values
        max_depth: maximum depth to be used for the base of the profile

    Returns:
        fig: plotly figure

    """
    import plotly.graph_objects as go

    base_depth = (prof[:, -2].min() - (prof[:, -2].max() - prof[:, -2].min())) if max_depth is None else max_depth
    base = np.ones_like(prof[:, -2]) * base_depth

    m_line, z_line = generate_terraincriteria_line(prof, depth=kp_depth, limit=limit)

    traces = [
        go.Scatter(
            x=prof[:, -1],
            y=base,
            line_color="rgba(161, 99, 18, 0.7)",
            fillcolor="rgba(161, 99, 18, 0.7)",
            mode="lines",
        ),
        go.Scatter(
            x=prof[:, -1],
            y=prof[:, -2],
            fill="tonexty",
            line_color="black",
            fillcolor="rgba(161, 99, 18, 0.7)",
            mode="lines",
        ),
        go.Scatter(
            x=m_line,
            y=z_line,
            line=dict(
                dash="dash",
                color="rgba(0, 0, 0, 0.7)",
                width=0.7,
            ),
            mode="lines",
            text=f"1:{int(limit)}-line",
        ),
    ]

    fig = go.Figure().add_traces(traces)
    coords_title = f"({prof[0, 0]:.1f}, {prof[0, 1]:.1f}) - ({prof[-1, 0]:.1f}, {prof[-1, 1]:.1f})"
    fig.update_layout(
        title=f"Profile @ {coords_title}",
        font=dict(size=10),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=20, autoexpand=True),
        dragmode="pan",
        width=600,
        height=210,
    )
    fig.update_annotations(font_size=10)
    return fig


def generate_terraincriteria_line(prof, limit=15, depth=0):
    """
    Generate a line with the terrain criteria (by default 1:15)
    Args:
        prof: numpy array with X, Y, Z, M values
        limit: Limit in vertical/horizontal ratio (1:limit) to be used for the terrain criteria
        depth: depth to be used for the line

    Returns:
        m: numpy array with the M values
        z: numpy array with the Z values
    """
    z = prof[:, -2]
    m = prof[:, -1]
    i_min, z_min = np.argmin(z), z.min()
    m_0 = m[i_min]
    m_line = np.abs(m - m_0)
    z_line = z_line = z_min + m_line / limit - depth
    z_line[z_line > prof[:, -2]] = np.nan

    return m, z_line


def polygonize_results(
    result_array: np.ndarray, dem_profile: rasterio.profiles.Profile | dict, field="value", threshold_value=1
):
    """Polygonize a raster array into a GeoDataFrame.

    Converts pixels with values greater than or equal to the threshold into
    vector polygons.

    Args:
        result_array: Array with the results to polygonize.
        dem_profile: Rasterio profile of the raster.
        field: Field name for the polygonized results.
        threshold_value: Minimum value for polygonization.

    Returns:
        GeoDataFrame with the polygonized results.
    """

    raster_transform = dem_profile["transform"]
    raster_crs = dem_profile["crs"]

    results_binary = (result_array >= threshold_value).astype("int16")

    results = (
        {"properties": {"id": i, field: int(v)}, "geometry": s}
        for i, (s, v) in enumerate(shapes(results_binary, mask=None, transform=raster_transform))
    )
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster[field] >= threshold_value]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(raster_crs)
    return gpd_polygonized_raster


def rasterize_shape(
    shapes: gpd.GeoDataFrame | list[BaseGeometry], dem_profile: rasterio.profiles.Profile | dict
) -> np.ndarray:
    """Rasterize geometries into a binary numpy array.

    Args:
        shapes: Geometries as a GeoDataFrame or list of shapely geometries.
        dem_profile: Rasterio profile (must contain 'height', 'width', 'transform').

    Returns:
        Binary numpy array with 1 where geometries are present, 0 elsewhere.
    """
    dem_height = dem_profile["height"]
    dem_width = dem_profile["width"]
    dem_transform = dem_profile["transform"]
    if isinstance(shapes, gpd.GeoDataFrame):
        geom = [shapes_ii for shapes_ii in shapes.geometry]
    elif all(isinstance(shape, BaseGeometry) for shape in shapes):
        geom = shapes
    else:
        raise ValueError("shapes must be either a GeoDataFrame or a list with geometries")

    rasterized = rasterize(
        geom,
        out_shape=(dem_height, dem_width),
        fill=0,
        out=None,
        transform=dem_transform,
        all_touched=True,
        default_value=1,
        dtype=None,
    )

    return rasterized


def get_msml_mask(bounds: tuple, results_offset=100) -> gpd.GeoDataFrame:
    """
    Get the MSML mask as an array for the given bounds
    Args:
        bounds: tuple with the bounds (xmin, ymin, xmax, ymax)

    Returns:
        mask: geopandas dataframe if dem_profile is None
    """

    mask_msml = get_maringrense(bounds, "msml", results_offset)
    mask_aumg = get_maringrense(bounds, "area_under_mg", results_offset)
    if mask_msml.empty and mask_aumg.empty:
        return gpd.GeoDataFrame(geometry=[])
    mask_gpd = gpd.GeoDataFrame(pd.concat([mask_msml, mask_aumg], ignore_index=True))
    return gpd.clip(mask_gpd, bounds).dissolve()


def get_maringrense(bounds, layer, results_offset=100):
    """
    Retrieves the MarinGrense data within the specified bounds and layer.

    Args:
        bounds (tuple): The bounding box coordinates (xmin, ymin, xmax, ymax).
        layer (str): The layer name to query. Valid options are "msml" and "area_under_mg".
        results_offset (int, optional): The number of results to offset in each request. Defaults to 100.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the MarinGrense data.

    """

    xmin, ymin, xmax, ymax = bounds

    layer_dict = {"msml": 7, "area_under_mg": 8}
    layer_nr = layer_dict[layer]

    url = f"https://gis3.nve.no/map/rest/services/Mapservices/MarinGrense/MapServer/{layer_nr}/query"

    params = {
        "geometry": f"xmin:{xmin},ymin:{ymin},xmax:{xmax},ymax:{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "f": "geojson",
    }

    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])

    while data.get("exceededTransferLimit"):
        params["resultOffset"] = params.get("resultOffset", 0) + results_offset
        response = requests.get(url, params=params)
        data = response.json()
        features.extend(data.get("features", []))
    if len(features) == 0:
        return gpd.GeoDataFrame(geometry=[])
    return gpd.GeoDataFrame.from_features(features).set_crs(4326).to_crs(25833)


def check_maringrense():
    xmin, ymin, xmax, ymax = 265122.0, 6648110.0, 266151.0, 6648761.0
    layer_dict = {"msml": 7, "area_under_mg": 8}

    params = {
        "geometry": f"xmin:{xmin},ymin:{ymin},xmax:{xmax},ymax:{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "f": "geojson",
    }

    for layer in ["msml", "area_under_mg"]:
        layer_nr = layer_dict[layer]

        url = f"https://gis3.nve.no/map/rest/services/Mapservices/MarinGrense/MapServer/{layer_nr}/query"

        try:
            response = requests.get(url, params=params)
            data = response.json()
        except requests.JSONDecodeError:
            return False
        if data.get("error") is not None:
            return False
        else:
            return True


def get_ar5_mask(bounds, results_offset=100):
    """
    Retrieves the AR5 data for "grunnlendt" and "fjell i dagen" within the specified bounds.

    """

    xmin, ymin, xmax, ymax = bounds

    url = "https://gis3.nve.no/map/rest/services/Mapservices/FKB/MapServer/2/query"

    params = {
        "geometry": f"xmin:{xmin},ymin:{ymin},xmax:{xmax},ymax:{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "outFields": "grunnforhold",
        "f": "geojson",
    }

    response = requests.get(url, params=params)
    # print(response.url)
    data = response.json()
    features = data.get("features", [])

    while data.get("exceededTransferLimit"):
        params["resultOffset"] = params.get("resultOffset", 0) + results_offset
        response = requests.get(url, params=params)
        data = response.json()
        features.extend(data.get("features", []))
    if len(features) == 0:
        return gpd.GeoDataFrame(geometry=[])
    gdf = gpd.GeoDataFrame.from_features(features).set_crs(4326).to_crs(25833)
    gdf = gdf.query("grunnforhold in (42, 43)").copy()
    # 42: fjell i dagen, 43: grunnlendt (https://register.geonorge.no/sosi-kodelister/fkb/ar5/5.0/arealressursgrunnforhold)
    gdf = gdf.clip(bounds)

    return gdf


def get_clipping_mask(bounds, msml=True, ar5=True):
    if not msml and not ar5:
        return None
    mask_msml = get_msml_mask(bounds) if msml else None
    if not ar5:
        return mask_msml
    mask_ar5 = get_ar5_mask(bounds) if ar5 else None

    base_mask = mask_msml if mask_msml is not None else gpd.GeoDataFrame(geometry=[box(*bounds)], crs=25833)
    try:
        mask = base_mask.overlay(mask_ar5, how="difference")
    except Exception as e:
        print(e)
        print(len(mask_ar5))
        print(len(base_mask))
        raise
    return mask


def modify_release_mask(
    release_mask, no_release_mask: gpd.GeoDataFrame = None, sup_release_mask: gpd.GeoDataFrame = None
):
    """
    Modify the release mask by removing the no release areas and adding the supplementary release areas
    Args:
        release_mask: release mask as a geopandas dataframe
        no_release_mask: no release mask as a geopandas dataframe
        sup_release_mask: supplementary release mask as a geopandas dataframe

    Returns:
        release_mask: modified release mask
    """

    if no_release_mask is not None:
        release_mask = gpd.GeoDataFrame(
            geometry=release_mask.dissolve().difference(no_release_mask.dissolve()), crs=release_mask.crs
        )

    if sup_release_mask is not None:
        release_mask = gpd.GeoDataFrame(
            geometry=release_mask.dissolve().union(sup_release_mask.dissolve()), crs=release_mask.crs
        )
    return release_mask


def generate_windows(custom_raster: str):
    """
    Generate windows from a raster in the same way get_hoydedata does.
    Args:
        custom_raster: path to the raster

    Returns:
        dictionary with the windows, transforms, dem arrays and the profile of the raster
    """
    blockxsize, blockysize = 640, 640

    with rasterio.open(custom_raster) as src:
        dataset_profile = src.profile
        full_array = src.read(1)
        dataset_transform = src.transform

        windows_dems = []
        windows_transforms = []

        block_shapes = np.array(src.block_shapes).squeeze()

        if np.any(block_shapes == 1):
            with tempfile.TemporaryDirectory() as temp_folder:
                dataset_profile.update(blockxsize=blockxsize, blockysize=blockysize, tiled=True)
                new_raster = os.path.join(temp_folder, "reblocked")
                with rasterio.open(new_raster, "w+", **dataset_profile) as out:
                    out.write(full_array, indexes=1)
                return generate_windows(new_raster)

        windows = [window for _, window in src.block_windows()]

        for window in windows:
            data = src.read(1, window=window)
            windows_dems.append(data)
            windows_transforms.append(rasterio.windows.transform(window, dataset_transform))

    return {
        "windows_dem_arrays": windows_dems,
        "windows_transforms": windows_transforms,
        "windows": windows,
        "profile": dataset_profile,
        "full_array": full_array,
    }


def convert_lines_to_gpd(lines: set) -> gpd.GeoDataFrame:
    """
    Function to convert a line into a geopandas dataframe.
    It is used to save the profiles that are being plotted.

    Args:
        lines (set): a set object with the lines taken from the map.

    Returns:
        gpd.GeoDataFrame: dataframe in wgs coordinate system
    """
    line_shapes = []
    line_id = []
    for index, line in enumerate(lines):
        line_np = np.array(line)
        line_shapes.append(LineString(line_np))
        line_id.append(index)

    gdf = gpd.GeoDataFrame(index=line_id, crs="epsg:4326", geometry=line_shapes)

    return gdf


def generate_fake_slope(
    base_length, base_elevation, terrace_length, terrace_elevation, slope_ratio, xmin=2e5, ymax=6e6
):

    resolution = 1
    # Calculate the length of the flat base and the sloped region
    slope_length = (terrace_elevation - base_elevation) // slope_ratio
    width = base_length + slope_length + terrace_length

    # Create the flat base
    base_array = np.full((int(width), int(base_length)), base_elevation)

    # Create the slope
    slope_array = np.linspace(base_elevation, terrace_elevation, int(slope_length))
    slope_array = np.tile(slope_array, (int(width), 1))

    # Create the terrace
    terrace_array = np.full((int(width), int(terrace_length)), terrace_elevation)

    # Concatenate the three sections to create the full terrain
    dem = np.hstack((base_array, slope_array, terrace_array))

    transform = rasterio.transform.from_origin(xmin, ymax, resolution, resolution)

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:25833",
        "transform": transform,
        "nodata": -9999,
    }

    return dem, profile


# --- Geometry utils for profile_retrogression ---
# TODO: Implementations to be ported from external project.


def _clip_single_line(base_line, crossing_line):
    """Helper to clip a single LineString at intersections with a base line.

    Args:
        base_line: The reference line used for clipping.
        crossing_line: The line to be clipped.

    Returns:
        The clipped geometry (first segment before crossing).
    """
    if not crossing_line.crosses(base_line):
        return crossing_line

    intersection = base_line.intersection(crossing_line)
    if intersection.is_empty:
        return crossing_line

    crossing_coords = list(crossing_line.coords)

    if intersection.geom_type == "LineString":
        coords = list(intersection.coords)
        intersection = MultiPoint([Point(coords[0]), Point(coords[-1])])

    elif intersection.geom_type == "MultiLineString":
        points = []
        for line in intersection.geoms:
            coords = list(line.coords)
            points.extend([Point(coords[0]), Point(coords[-1])])
        intersection = MultiPoint(points)

    elif intersection.geom_type == "GeometryCollection":
        points = [geom for geom in intersection.geoms if geom.geom_type == "Point"]
        if points:
            intersection = MultiPoint(points) if len(points) > 1 else points[0]
        else:
            return crossing_line

    elif intersection.geom_type == "Point":
        # CHECK: Is the point too close to the endpoints?
        int_point = intersection
        dist_to_start = int_point.distance(Point(crossing_coords[0]))
        dist_to_end = int_point.distance(Point(crossing_coords[-1]))

        # If intersection is too close to endpoints, split won't work
        tolerance = 0.01  # 1cm tolerance
        if dist_to_start < tolerance:
            return crossing_line
        if dist_to_end < tolerance:
            return crossing_line

    elif intersection.geom_type == "MultiPoint":
        pass
    else:
        return crossing_line

    # Now split using the point(s)
    try:
        segments = split(crossing_line, intersection)

        if hasattr(segments, "geoms"):
            # If split produced only 1 segment, something went wrong
            if len(segments.geoms) == 1:
                # Try with a small buffer to "snap" the point to the line
                buffered_intersection = intersection.buffer(0.001)
                try:
                    segments_retry = split(crossing_line, buffered_intersection)
                    if hasattr(segments_retry, "geoms") and len(segments_retry.geoms) > 1:
                        return segments_retry
                except Exception:
                    pass

        return segments

    except Exception:
        logger.debug("Split failed in _clip_single_line", exc_info=True)
        return crossing_line


def clip_crossing_lines(base_line, crossing_line):
    """Clip a line at its intersections with a base line.

    Handles both LineString and MultiLineString inputs. Returns the
    first segment before the crossing point.

    Args:
        base_line: LineString or MultiLineString used as the cutting reference.
        crossing_line: LineString or MultiLineString to be clipped.

    Returns:
        The clipped geometry segment, or the original if no crossing found.
    """
    if isinstance(base_line, MultiLineString):
        base_line = linemerge(base_line)

    # Handle MultiLineString crossing_line
    if isinstance(crossing_line, MultiLineString):
        all_segments = []

        for i, part in enumerate(crossing_line.geoms):
            if isinstance(part, LineString):
                result = _clip_single_line(base_line, part)

                if hasattr(result, "geoms"):  # GeometryCollection
                    all_segments.extend(result.geoms)
                else:  # Single LineString
                    all_segments.append(result)

        if len(all_segments) > 1:
            return all_segments[0]  # GeometryCollection(all_segments)
        elif len(all_segments) == 1:
            return all_segments[0]
        else:
            return crossing_line

    # Handle single LineString
    else:
        result = _clip_single_line(base_line, crossing_line)
        if type(result) is GeometryCollection:
            result = result.geoms[0]
        return result


def create_one_sided_sections_along_line(
    geometry,
    spacing=5,
    length=100,
    side="right",
    create_fans=False,
):
    """Create equally spaced perpendicular lines on one side of a LineString.

    Generates perpendicular cross-sections at regular intervals along the
    input line. Optionally creates fan-shaped profiles at the start and end.

    Args:
        geometry: LineString or MultiLineString input line.
        spacing: Distance in meters between perpendicular lines.
        length: Length of each perpendicular line in meters.
        side: Side to create perpendiculars ("left" or "right").
        create_fans: Whether to include fan-profiles at start/end of line.

    Returns:
        List of perpendicular LineString geometries.
    """
    num_profiles_start_end = 5
    tan_start_type = "quarter_line"

    def get_perpendicular_at_point(line_geom, distance_along, perp_length, side_multiplier):
        point = line_geom.interpolate(distance_along)

        # Find the nearest segment to get direction
        coords = list(line_geom.coords)
        min_dist = float("inf")
        best_segment = None

        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            dist = segment.distance(point)
            if dist < min_dist:
                min_dist = dist
                best_segment = segment

        if best_segment is None:
            return None

        # Get direction vector of the segment
        seg_coords = list(best_segment.coords)
        dx = seg_coords[1][0] - seg_coords[0][0]
        dy = seg_coords[1][1] - seg_coords[0][1]

        # Normalize direction vector
        length_seg = np.sqrt(dx**2 + dy**2)
        if length_seg == 0:
            return None

        dx_norm = dx / length_seg
        dy_norm = dy / length_seg

        # Get perpendicular vector (rotate 90 degrees)
        perp_dx = -dy_norm * side_multiplier
        perp_dy = dx_norm * side_multiplier

        # Create perpendicular line
        start_x = point.x
        start_y = point.y
        end_x = point.x + perp_dx * perp_length
        end_y = point.y + perp_dy * perp_length

        return LineString([(start_x, start_y), (end_x, end_y)])

    def generate_fan(point, tangent, is_start, num_profiles):
        # Normalize tangent
        norm = np.sqrt(tangent[0] ** 2 + tangent[1] ** 2)
        if norm == 0:
            return []
        tan_u = np.array([tangent[0] / norm, tangent[1] / norm])

        # Calculate perpendicular (0 degrees) matching create_perpendicular_lines logic
        perp = np.array([-tan_u[1] * side_multiplier, tan_u[0] * side_multiplier])

        # Target vector (90 degrees)
        # If start, target is -tangent (backward)
        # If end, target is tangent (forward)
        target = -tan_u if is_start else tan_u

        # Determine rotation direction from perp to target
        # Cross product (2D): x1*y2 - y1*x2
        cp = perp[0] * target[1] - perp[1] * target[0]

        # If cp > 0, target is CCW from perp. Angle is +90.
        angle_total = np.pi / 2 if cp > 0 else -np.pi / 2

        angles = np.linspace(0, angle_total, num_profiles)

        fan_lines = []
        for angle in angles:
            c, s = np.cos(angle), np.sin(angle)
            # Rotate perp vector
            vx = perp[0] * c - perp[1] * s
            vy = perp[0] * s + perp[1] * c

            end_pt = (point[0] + vx * length, point[1] + vy * length)
            fan_lines.append(LineString([point, end_pt]))

        return fan_lines

    # Handle MultiLineString by converting to single LineString
    if isinstance(geometry, MultiLineString):
        # Merge all parts into a single line
        coords = []
        for geom in geometry.geoms:
            coords.extend(list(geom.coords))
        line_geom = LineString(coords)
    else:
        line_geom = geometry
        coords = geometry.coords

    # Determine side multiplier
    side_multiplier = 1 if side == "right" else -1

    # Get total length of the line
    total_length = line_geom.length

    distances = list(np.arange(spacing, total_length, spacing))
    distances = [0.0] + distances + [total_length]
    distances_clean = [distances[0]]
    for d in distances[1:]:
        if abs(d - distances_clean[-1]) > 0.1:
            distances_clean.append(d)
    distances = distances_clean

    # Create perpendicular lines
    perpendiculars = []
    for dist in distances:
        if dist == 0:
            if tan_start_type == "last_segment":
                tan_start = np.array(coords[1]) - np.array(coords[0])
            elif tan_start_type == "half_line":
                mid_point = line_geom.interpolate(total_length / 2)
                tan_start = np.array(mid_point.coords[0]) - np.array(coords[0])
            elif tan_start_type == "quarter_line":
                quarter_point = line_geom.interpolate(total_length / 4)
                tan_start = np.array(quarter_point.coords[0]) - np.array(coords[0])

            if create_fans:
                perp = generate_fan(coords[0], tan_start, is_start=True, num_profiles=num_profiles_start_end)
                perpendiculars.extend(perp)
            else:
                perp = get_perpendicular_at_point(line_geom, 0, length, side_multiplier)
                if perp:
                    perpendiculars.append(perp)

        elif dist >= total_length:
            if tan_start_type == "last_segment":
                tan_end = np.array(coords[-1]) - np.array(coords[-2])
            elif tan_start_type == "half_line":
                mid_point = line_geom.interpolate(total_length / 2)
                tan_end = np.array(coords[-1]) - np.array(mid_point.coords[0])
            elif tan_start_type == "quarter_line":
                quarter_point = line_geom.interpolate(3 * total_length / 4)
                tan_end = np.array(coords[-1]) - np.array(quarter_point.coords[0])

            if create_fans:
                perp = generate_fan(coords[-1], tan_end, is_start=False, num_profiles=num_profiles_start_end)
                perpendiculars.extend(perp)
            else:
                perp = get_perpendicular_at_point(line_geom, total_length, length, side_multiplier)
                if perp:
                    perpendiculars.append(perp)

        else:
            # Normal case - point is on the line
            perp = get_perpendicular_at_point(line_geom, dist, length, side_multiplier)

            if perp:
                perpendiculars.append(perp)

    return perpendiculars


def generate_points_along_lines(gdf: gpd.GeoDataFrame, max_distance: float) -> gpd.GeoDataFrame:
    """Generate points at regular intervals along line geometries.

    Args:
        gdf: GeoDataFrame containing line geometries.
        max_distance: Maximum distance in meters between generated points.

    Returns:
        GeoDataFrame containing the generated points with original attributes.
    """

    points_list = []

    for idx, row in gdf.iterrows():
        geom = row.geometry

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            line_length = line.length

            if line_length == 0:
                continue

            # Calculate number of points needed
            num_points = int(line_length / max_distance) + 1

            # Generate points at regular intervals
            for i in range(num_points + 1):
                distance = min(i * max_distance, line_length)
                point = line.interpolate(distance)

                # Create new row with point geometry and original attributes
                new_row = row.copy()
                new_row.geometry = point
                points_list.append(new_row)

    # Create new GeoDataFrame with points
    points_gdf = gpd.GeoDataFrame(points_list, crs=gdf.crs).reset_index(drop=True)

    return points_gdf


def extract_elevation_values_for_points(
    point_array: np.ndarray, elevation_array: np.ndarray, raster_profile: dict
) -> np.ndarray:
    """Extract elevation values from a DEM for given point coordinates.

    Args:
        point_array: Array of (x, y) coordinates.
        elevation_array: The DEM as a 2D numpy array.
        raster_profile: Rasterio profile dict (must contain 'transform').

    Returns:
        Array of elevation values, with NaN for out-of-bounds points.
    """
    if point_array.shape == (2,):
        points_xy = np.expand_dims(point_array, 0)
    else:
        points_xy = point_array.copy()

    transform = raster_profile["transform"]
    height, width = elevation_array.shape

    # Convert coordinates to pixel indices
    pixel_indices = np.array([rasterio.transform.rowcol(transform, point[0], point[1]) for point in points_xy])

    # Extract elevation values, using NaN for out-of-bounds points
    elevation_values = np.full(len(pixel_indices), np.nan, dtype=float)
    for i, idx in enumerate(pixel_indices):
        if 0 <= idx[0] < height and 0 <= idx[1] < width:
            elevation_values[i] = elevation_array[idx[0], idx[1]]

    return elevation_values


def create_terrain_profile(
    line: BaseGeometry | gpd.GeoDataFrame, dem_array: np.ndarray, profile: dict, resolution: float = 5.0
) -> tuple[list[float], np.ndarray]:
    """Extract a terrain profile (distances and elevations) along a line from a DEM.

    Args:
        line: LineString geometry or GeoDataFrame with a line geometry.
        dem_array: The DEM as a 2D numpy array.
        profile: Rasterio profile dict (must contain 'transform' and 'crs').
        resolution: Sampling resolution in meters along the line.

    Returns:
        Tuple of (distances, elevations) where distances is a list of cumulative
        distances and elevations is an array of corresponding elevation values.
    """
    if isinstance(line, gpd.GeoDataFrame):
        gdf = line.to_crs(profile["crs"])
    else:
        gdf = gpd.GeoDataFrame(geometry=[line], crs=profile["crs"])
    points_xy = generate_points_along_lines(gdf, max_distance=resolution).get_coordinates().values
    elevations = extract_elevation_values_for_points(points_xy, dem_array, profile)

    distances = [0]
    for i in range(1, len(points_xy)):
        prev_point = points_xy[i - 1]
        curr_point = points_xy[i]
        dist = np.sqrt((curr_point[0] - prev_point[0]) ** 2 + (curr_point[1] - prev_point[1]) ** 2)
        distances.append(distances[-1] + dist)

    return distances, elevations


def generate_envelope_around_points(points: list[Point]) -> BaseGeometry:
    """Generate a convex hull polygon around a set of points.

    Args:
        points: List of shapely Point geometries.

    Returns:
        A shapely polygon (convex hull) enclosing all points.
    """
    return _generate_envelope_around_points_brute_force(points)


def _generate_envelope_around_points_brute_force(points: list[Point]) -> BaseGeometry:
    """Compute a convex hull from a collection of points."""
    multi_point = MultiPoint(points)
    return multi_point.convex_hull


def plot_geometries(
    geometries: list[BaseGeometry] | BaseGeometry, ax=None, color: str = "black", alpha: float = 1
):
    """Plot shapely geometries on a matplotlib axes.

    Args:
        geometries: Single geometry or list of geometries to plot.
        ax: Matplotlib axes. If None, creates a new figure.
        color: Color for the plotted geometries.
        alpha: Transparency level.

    Returns:
        The matplotlib axes with the plotted geometries.
    """
    if not isinstance(geometries, list):
        geom_list = [geometries]
    else:
        geom_list = geometries
    if ax is None:
        ax = gpd.GeoDataFrame(geometry=geom_list).plot(color=color, alpha=alpha)
    else:
        gpd.GeoDataFrame(geometry=geom_list).plot(ax=ax, color=color, alpha=alpha)
    return ax
