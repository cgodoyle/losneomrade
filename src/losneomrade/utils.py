import os
import tempfile
import time
import warnings
import requests
from urllib.request import urlopen

import rasterio
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from rasterio import MemoryFile
from rasterio.features import rasterize, shapes
from scipy.spatial import distance_matrix
import pandas as pd


warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

HOYDEDATA_LAYER = "NHM_DTM_25833"


def get_hoydedata(bounds: tuple, layer: str = HOYDEDATA_LAYER, res: int = 5, nodata: int = -9999, max_retries=5) -> dict:
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
        except Exception as e:
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

                    windows_transforms.append(
                        rasterio.windows.transform(window, dataset.transform)
                    )

    except Exception as e:
        print(request_url)
        print("Error (Probably area requested is too big/small or høydedata is down)")
        raise

    return {
        "windows_dem_arrays": windows_dems,
        "windows_transforms": windows_transforms,
        "windows": windows,
        "profile": dataset_profile,
        "full_array": full_array
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
    gdf = gpd.GeoDataFrame(
        index=[0], crs=f"epsg:{int(fra_crs)}", geometry=[line]
    ).to_crs(epsg=to_crs)
    linea2 = gdf.iloc[0].geometry
    n_points = int(max(linea2.length / 5, 5))
    new_points = [
        linea2.interpolate(i / float(n_points - 1), normalized=True)
        for i in range(n_points)
    ]
    points_coords = np.array(
        [[pp.coords.xy[0][0], pp.coords.xy[1][0]] for pp in new_points]
    )

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
    tif_bytes = urlopen(request_url).read()

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

    cum_dist = np.cumsum(
        np.sqrt(
            np.sum(
                (np.r_[[[0, 0]], np.diff(points_coords, axis=0)[:, :2]]) ** 2, axis=1
            )
        )
    )

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
    base_depth = (prof[:, -2].min() - (prof[:, -2].max() - prof[:, -2].min())) if max_depth is None else max_depth
    base = np.ones_like(prof[:, -2]) * base_depth

    m_line, z_line = generate_terraincriteria_line(prof, depth=kp_depth, limit=limit)

    traces = [go.Scatter(x=prof[:, -1], y=base,
                         line_color="rgba(161, 99, 18, 0.7)", fillcolor="rgba(161, 99, 18, 0.7)",
                         mode="lines", ),
              go.Scatter(x=prof[:, -1], y=prof[:, -2],
                         fill="tonexty", line_color="black", fillcolor="rgba(161, 99, 18, 0.7)",
                         mode="lines", ),
              go.Scatter(x=m_line, y=z_line, line=dict(dash='dash', color="rgba(0, 0, 0, 0.7)", width=0.7, ),
                         mode="lines", text=f"1:{int(limit)}-line"),
              ]

    fig = go.Figure().add_traces(traces)
    coords_title = f"({prof[0, 0]:.1f}, {prof[0, 1]:.1f}) - ({prof[-1, 0]:.1f}, {prof[-1, 1]:.1f})"
    fig.update_layout(title=f"Profile @ {coords_title}",
                      font=dict(size=10),
                      showlegend=False,
                      margin=dict(l=0, r=0, t=20, b=20, autoexpand=True),
                      dragmode="pan", width=600, height=210, )
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


def polygonize_results(result_array: np.ndarray, dem_profile: rasterio.profiles.Profile,
                       field="value", threshold_value=1):
    """
    Polygonize an array whose values are greather tha the given threshold and save the results as a 
    geopandas dataframe with the given field name.

    Args:
        result_array: array with the results
        dem_profile: profile of the dem
        field: field name to be used for the polygonized results
        threshold_value: value to be used as threshold for the polygonization

    Returns:
        gpd_polygonized_raster: geopandas dataframe with the polygonized results
    """

    raster_transform = dem_profile['transform']
    raster_crs = dem_profile['crs']

    results_binary = (result_array >= threshold_value).astype(int)

    results = ({"properties": {"id": i, field: int(v)}, "geometry": s}
               for i, (s, v) in enumerate(shapes(results_binary, mask=None, transform=raster_transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster[field] >= threshold_value]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(raster_crs)
    return gpd_polygonized_raster


def rasterize_shape(release_shp, dem_profile: rasterio.profiles.Profile) -> np.ndarray:
    """
    Rasterize the release area
    Args:
        release_shp: release area as a geopandas dataframe
        dem_profile: profile of the dem

    Returns:
        rasterized: rasterized release area as a numpy array
    """
    dem_height = dem_profile['height']
    dem_width = dem_profile['width']
    dem_transform = dem_profile['transform']

    geom = [shapes_ii for shapes_ii in release_shp.geometry]

    rasterized = rasterize(geom,
                           out_shape=(dem_height, dem_width),
                           fill=0,
                           out=None,
                           transform=dem_transform,
                           all_touched=True,
                           default_value=1,
                           dtype=None)

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
    mask_gpd = gpd.GeoDataFrame(pd.concat([mask_msml, mask_aumg], ignore_index=True)).set_crs(4326).to_crs(25833)
    return gpd.clip(mask_gpd,bounds).dissolve()


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
        "f": "geojson"
    }

    response = requests.get(url, params=params)
    data = response.json()
    features = data.get("features", [])

    while data.get("exceededTransferLimit"):
        params["resultOffset"] = params.get("resultOffset", 0) + results_offset
        response = requests.get(url, params=params)
        data = response.json()
        features.extend(data.get("features", []))

    return gpd.GeoDataFrame.from_features(features)


def modify_release_mask(release_mask, no_release_mask: gpd.GeoDataFrame = None, sup_release_mask: gpd.GeoDataFrame = None):
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
            geometry=release_mask.dissolve().difference(no_release_mask.dissolve()), crs=release_mask.crs)

    if sup_release_mask is not None:
        release_mask = gpd.GeoDataFrame(
            geometry=release_mask.dissolve().union(sup_release_mask.dissolve()), crs=release_mask.crs)
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
            windows_transforms.append(
                rasterio.windows.transform(window, dataset_transform)
            )

    return {
        "windows_dem_arrays": windows_dems,
        "windows_transforms": windows_transforms,
        "windows": windows,
        "profile": dataset_profile,
        "full_array": full_array
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
    from shapely.geometry import LineString
    line_shapes = []
    line_id = []
    for index, line in enumerate(lines):
        line_np = np.array(line)
        line_shapes.append(LineString(line_np))
        line_id.append(index)

    gdf = gpd.GeoDataFrame(index=line_id, crs="epsg:4326", geometry=line_shapes)

    return gdf


def generate_fake_slope(base_length, base_elevation, terrace_length, terrace_elevation, slope_ratio, xmin=2e5, ymax=6e6):

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
        'driver': 'GTiff',
        'height': dem.shape[0],
        'width': dem.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:25833',
        'transform': transform,
        'nodata': -9999,
    }

    return dem, profile
