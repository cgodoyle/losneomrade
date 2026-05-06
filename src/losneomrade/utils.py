import logging
import os
import tempfile
import warnings

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from scipy.spatial import distance_matrix
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, split

from .config import settings
from .hoydedata import get_hoydedata  # noqa: F401
from .types import DataFrameLike

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

HOYDEDATA_LAYER = settings.hoydedata.layer


def dem_coordinates(dem_array: np.ndarray, dem_transform: rasterio.transform.Affine) -> np.ndarray:
    """Get coordinates for a DEM window.

    Args:
        dem_array: Window elevation array.
        dem_transform: Transform for the DEM window.

    Returns:
        Numpy array with the coordinates as ``(x, y, z)``.
    """
    height, width = dem_array.shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(dem_transform, rows, cols)
    x_dem = np.array(xs)
    y_dem = np.array(ys)
    coords = np.c_[x_dem.flatten(), y_dem.flatten(), dem_array.flatten()]
    return coords


def compute_slope(coords: np.ndarray, points: np.ndarray, h_min: float = 5, nodata: int = -9999) -> np.ndarray:
    """Compute slopes for DEM coordinates relative to source points.

    Args:
        coords: DEM window coordinates.
        points: Source point coordinates.
        h_min: Minimum height difference where slopes are calculated.
        nodata: Value assigned to pixels with no data.

    Returns:
        Array with slopes for each DEM coordinate.
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
    """Compute slopes for DEM coordinates relative to source points in chunks.

    Args:
        coords: DEM window coordinates.
        points: Source point coordinates.
        h_min: Minimum height difference where slopes are calculated.
        nodata: Value assigned to pixels with no data.
        chunk_size: Number of points to process in each batch.

    Returns:
        Array with slopes for each DEM coordinate.
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
    """Set elevation values for the given ``x, y`` points.

    Args:
        points_xy: Array with ``x, y`` coordinates for the points.
        window_data: DEM results from calling ``get_hoydedata``.

    Returns:
        Numpy array with ``x, y, z`` coordinates.
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


def generate_plotly_profile(
    prof: np.ndarray, max_depth: float | None = None, kp_depth: float = 0, limit: float = 15
) -> object:
    """Generate a Plotly figure for a terrain profile.

    Args:
        prof: Numpy array with ``X, Y, Z, M`` values.
        max_depth: Maximum depth to use for the base of the profile.
        kp_depth: Depth offset to apply to the terrain criteria line.
        limit: Terrain criteria ratio expressed as ``1:limit``.

    Returns:
        Plotly figure for the profile.
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


def generate_terraincriteria_line(
    prof: np.ndarray, limit: float = 15, depth: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a terrain criteria line for a profile.

    Args:
        prof: Numpy array with ``X, Y, Z, M`` values.
        limit: Terrain criteria ratio expressed as ``1:limit``.
        depth: Depth offset to apply to the line.

    Returns:
        Tuple with the profile ``M`` values and generated ``Z`` values.
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
    result_array: np.ndarray,
    dem_profile: rasterio.profiles.Profile | dict,
    field: str = "value",
    threshold_value: int | float = 1,
) -> gpd.GeoDataFrame:
    """Polygonize a raster array into a GeoDataFrame.

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
    """Rasterize geometries into a binary array.

    Args:
        shapes: Geometries as a GeoDataFrame or list of shapely geometries.
        dem_profile: Rasterio profile containing ``height``, ``width``, and ``transform``.

    Returns:
        Binary array with ``1`` where geometries are present and ``0`` elsewhere.
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


# Backward-compat re-exports from masks module
from losneomrade.masks import get_msml_mask as get_msml_mask  # noqa: E402
from losneomrade.masks import modify_release_mask as modify_release_mask  # noqa: E402


def generate_windows(custom_raster: str) -> dict:
    """Generate raster windows in the same way as ``get_hoydedata``.

    Args:
        custom_raster: Path to the raster.

    Returns:
        Dictionary with raster windows, transforms, DEM arrays, and profile.
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
    """Convert lines into a GeoDataFrame.

    Args:
        lines: Set with line coordinate sequences taken from the map.

    Returns:
        GeoDataFrame in WGS84 containing the input lines.
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
    base_length: float,
    base_elevation: float,
    terrace_length: float,
    terrace_elevation: float,
    slope_ratio: float,
    xmin: float = 2e5,
    ymax: float = 6e6,
) -> tuple[np.ndarray, dict]:
    """Generate a synthetic slope DEM for testing.

    Args:
        base_length: Length of the flat base section.
        base_elevation: Elevation of the flat base section.
        terrace_length: Length of the upper terrace section.
        terrace_elevation: Elevation of the upper terrace section.
        slope_ratio: Vertical-to-horizontal slope ratio denominator.
        xmin: Origin ``x`` coordinate for the raster transform.
        ymax: Origin ``y`` coordinate for the raster transform.

    Returns:
        Tuple containing the generated DEM array and raster profile.
    """

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


def clip_crossing_lines(
    base_line: LineString | MultiLineString, crossing_line: LineString | MultiLineString
) -> BaseGeometry:
    """Clip a line at its intersections with a base line.

    Args:
        base_line: LineString or MultiLineString used as the cutting reference.
        crossing_line: LineString or MultiLineString to be clipped.

    Returns:
        Clipped geometry segment, or the original geometry if no crossing is found.
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
    geometry: LineString | MultiLineString,
    spacing: float = 5,
    length: float = 100,
    side: str = "right",
    create_fans: bool = False,
) -> list[LineString]:
    """Create equally spaced perpendicular lines on one side of a line.

    Args:
        geometry: LineString or MultiLineString input line.
        spacing: Distance in meters between perpendicular lines.
        length: Length of each perpendicular line in meters.
        side: Side to create perpendiculars on, either ``"left"`` or ``"right"``.
        create_fans: Whether to include fan profiles at the start and end of the line.

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


def generate_points_along_lines(gdf: gpd.GeoDataFrame, max_distance: float) -> DataFrameLike:
    """Generate points at regular intervals along line geometries.

    Args:
        gdf: GeoDataFrame containing line geometries.
        max_distance: Maximum distance in meters between generated points.

    Returns:
        GeoDataFrame containing generated points with the original attributes.
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
    """Extract elevation values from a DEM for point coordinates.

    Args:
        point_array: Array of ``(x, y)`` coordinates.
        elevation_array: DEM as a 2D numpy array.
        raster_profile: Raster profile dictionary containing ``transform``.

    Returns:
        Array of elevation values, with ``NaN`` for out-of-bounds points.
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
) -> tuple[list, np.ndarray]:
    """Create a terrain profile along a line from a DEM.

    Args:
        line: Line geometry or GeoDataFrame with a line geometry.
        dem_array: DEM as a 2D numpy array.
        profile: Raster profile dictionary containing ``transform`` and ``crs``.
        resolution: Sampling resolution in meters along the line.

    Returns:
        Tuple of cumulative distances and corresponding elevation values.
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
    """Generate a convex hull around a set of points.

    Args:
        points: List of shapely Point geometries.

    Returns:
        Shapely geometry enclosing all input points.
    """
    return _generate_envelope_around_points_brute_force(points)


def _generate_envelope_around_points_brute_force(points: list[Point]) -> BaseGeometry:
    """Compute a convex hull from a collection of points."""
    multi_point = MultiPoint(points)
    return multi_point.convex_hull


def plot_geometries(
    geometries: list[BaseGeometry] | BaseGeometry,
    ax: object | None = None,
    color: str = "black",
    alpha: float = 1,
) -> object:
    """Plot shapely geometries on a matplotlib axes.

    Args:
        geometries: Single geometry or list of geometries to plot.
        ax: Matplotlib axes. If ``None``, a new figure is created.
        color: Color for the plotted geometries.
        alpha: Transparency level.

    Returns:
        Matplotlib axes with the plotted geometries.
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
