import logging
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.profiles import Profile
from rasterio.transform import array_bounds
from shapely.geometry import Point, box

from .retrogression import landslide_retrogression
from .types import PathLike
from .utils import (
    clip_crossing_lines,
    create_one_sided_sections_along_line,
    create_terrain_profile,
    generate_envelope_around_points,
    rasterize_shape,
)
from .utils import (
    polygonize_results as polygonize_array,
)

logger = logging.getLogger(__name__)

type RasterProfile = dict[str, Any]


def _run_retrogression(
    dem_array: np.ndarray,
    stream_line: Any,
    dem_profile: RasterProfile,
    boundary_release: gpd.GeoDataFrame | None,
    slope_ratio: float = 1 / 15,
    min_height: float = 5,
    min_length: float = 75,
    mask_array: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Run the raster retrogression step constrained by a profile-derived release."""

    if boundary_release is None:
        release_mask = np.ones_like(dem_array, dtype=bool)
    else:
        release_mask = rasterize_shape(boundary_release, dem_profile)

    line_mask = rasterize_shape(gpd.GeoDataFrame(geometry=[stream_line.geometry]), dem_profile)
    combined_mask = (
        release_mask & mask_array if mask_array is not None and mask_array.shape == dem_array.shape else release_mask
    )

    release_array, _ = landslide_retrogression(
        dem_array,
        line_mask,
        dem_profile["transform"],
        initial_release_depth=stream_line.water_depth,
        min_slope=slope_ratio,
        min_height=min_height,
        min_length=min_length,
        mask=combined_mask,
        verbose=False,
    )
    raster_profile = Profile(**dem_profile)
    return polygonize_array(release_array, raster_profile, field="slope").to_crs(epsg=25833)


def depth_above_slope(
    distances: list[float] | np.ndarray,
    elevations: list[float] | np.ndarray,
    depth: float = 0,
    slope_ratio: float = 1 / 15,
) -> list[float]:
    """Calculate the vertical offset between terrain and a slope line."""

    slope_line = [elevations[0] - depth + slope_ratio * distance for distance in distances]
    return [elevations[i] - slope_line[i] for i in range(len(elevations))]


def retrogression_distance(
    distance: list[float] | np.ndarray,
    depth: list[float] | np.ndarray,
    ignore_first_meters: float = 0,
    debug: bool = False,
) -> float | None:
    """Return the first zero-crossing distance for a depth profile."""

    if len(distance) == 0 or len(depth) == 0:
        return None

    start_idx = 0
    for index, current_distance in enumerate(distance):
        if current_distance >= ignore_first_meters:
            start_idx = index
            break

    if start_idx >= len(depth) - 1:
        return None

    for index in range(start_idx, len(depth) - 1):
        left_depth = depth[index]
        right_depth = depth[index + 1]

        if (left_depth <= 0 and right_depth > 0) or (left_depth > 0 and right_depth <= 0):
            if right_depth == left_depth:
                return distance[index]

            ratio = abs(left_depth) / abs(right_depth - left_depth)
            return distance[index] + ratio * (distance[index + 1] - distance[index])

    remaining_depth = depth[start_idx:]
    if len(remaining_depth) == 0:
        return None

    if all(value >= 0 for value in remaining_depth):
        if debug:
            logger.debug(
                "Warning: No zero crossing found and all depths are positive after "
                f"{ignore_first_meters}m. Returning max distance {max(distance)}.",
            )
        return max(distance)

    if all(value <= 0 for value in remaining_depth):
        if debug:
            logger.debug(
                "Warning: No zero crossing found and all depths are negative after "
                f"{ignore_first_meters}m. Returning ignore_first_meters parameter.",
            )
        return ignore_first_meters

    if any(np.isnan(remaining_depth)):
        index_nan = next(i for i, value in enumerate(remaining_depth) if np.isnan(value)) + start_idx
        if debug:
            logger.debug(
                "Warning: No zero crossing found and NaN values present after "
                f"{ignore_first_meters}m. Returning {distance[index_nan]}.",
            )
        return distance[index_nan]

    return None


def get_coordinates_for_retrogression_distance(
    profiles: list[Any],
    distance_list: list[float | None],
) -> list[tuple[float, float] | None]:
    """Interpolate the retrogression point along each profile line."""

    coordinates: list[tuple[float, float] | None] = []
    for profile, distance in zip(profiles, distance_list):
        if distance is None or distance <= 0:
            coordinates.append(None)
            continue

        point = profile.interpolate(distance)
        coordinates.append((point.x, point.y))

    return coordinates


def _resolve_profile_sampling(
    line_length: float,
    spacing_m: float | None,
    n_profiles: int | None,
    debug: bool = False,
) -> tuple[float, int]:
    """Resolve a consistent spacing/profile-count pair."""

    if spacing_m is None and n_profiles is None:
        n_profiles = 3
        if debug:
            logger.debug(f"Number of profiles not specified, defaulting to {n_profiles}")

    if spacing_m is None and n_profiles is not None:
        if n_profiles < 3:
            raise ValueError("Minimum number of profiles is 3")
        spacing_m = float(np.round(line_length / (n_profiles - 1), 2))
        if debug:
            logger.debug(f"Calculated spacing between profiles: {spacing_m:.2f} meters. line length = {line_length}")
        return spacing_m, n_profiles

    if spacing_m is None:
        raise ValueError("spacing_m could not be resolved")

    if n_profiles is None:
        n_profiles = min(max(int(np.floor(line_length / spacing_m)) - 1, 3), 20)
        spacing_m = float(np.round(line_length / (n_profiles - 1), 1))
        if debug:
            logger.debug(f"Calculated number of profiles: {n_profiles}")
            logger.debug(f"Adjusted spacing between profiles: {spacing_m:.2f} meters")

    return spacing_m, n_profiles


def _build_profiles(
    line: Any,
    spacing_m: float,
    profile_length_m: float,
    side: str,
    clip_to_sourceline: bool,
) -> list[Any]:
    """Create one-sided terrain profiles for a source line."""

    profiles = create_one_sided_sections_along_line(line, spacing=spacing_m, length=profile_length_m, side=side)
    if not clip_to_sourceline:
        return profiles

    return [clip_crossing_lines(line, profile) for profile in profiles]


def _mask_dem_array(dem_array: np.ndarray, mask_array: np.ndarray | None) -> np.ndarray:
    """Apply a binary mask to the DEM when shapes are compatible."""

    if mask_array is None:
        return dem_array
    if mask_array.shape != dem_array.shape:
        logger.debug("Mask shape does not match DEM shape. Ignoring mask.")
        return dem_array
    return np.where(mask_array == 1, dem_array, np.nan)


def _line_vertex_points(line: Any) -> list[Point]:
    """Convert line vertices to point geometries for envelope generation."""

    if line.geom_type == "MultiLineString":
        return [Point(coord) for geometry in line.geoms for coord in geometry.coords]
    if line.geom_type == "LineString":
        return [Point(coord) for coord in line.coords]
    raise ValueError(f"Check line {line} since it is not a multi nor single linestring")


def retrogression_from_profiles(
    line: Any,
    dem_array: np.ndarray,
    dem_profile: RasterProfile,
    slope_ratio: float = 1 / 15,
    spacing_m: float | None = None,
    n_profiles: int | None = None,
    profile_length_m: float = 500,
    side: str = "right",
    depth_m: float = 0,
    tolerance_distance_m: float = 0,
    min_height_m: float = 5,
    max_height_m: float | None = 30,
    mask_array: np.ndarray | None = None,
    save_plots_path: PathLike | None = None,
    debug: bool = False,
    min_n_points_for_envelope: int = 3,
    clip_to_sourceline: bool = True,
) -> tuple[gpd.GeoDataFrame, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame] | None]:
    """Compute a retrogression envelope from side profiles along a stream line."""

    del save_plots_path

    if max_height_m is None:
        max_height_m = float(np.inf)

    calc_depth = depth_m + 0.001
    dem_bounds = array_bounds(dem_profile["height"], dem_profile["width"], dem_profile["transform"])
    resolution_m = float(np.mean(np.abs([dem_profile["transform"][0], dem_profile["transform"][4]])))

    if not line.intersects(box(*dem_bounds)):
        logger.error("The provided line does not intersect the DEM extent.")
        return gpd.GeoDataFrame(geometry=[], crs=25833), None

    spacing_m, _ = _resolve_profile_sampling(line.length, spacing_m, n_profiles, debug=debug)

    try:
        profiles = _build_profiles(
            line,
            spacing_m=spacing_m,
            profile_length_m=profile_length_m,
            side=side,
            clip_to_sourceline=clip_to_sourceline,
        )
    except Exception as exc:
        logger.exception(
            f"Error creating profiles for line length={line.length}, spacing_m={spacing_m}, "
            f"profile_length_m={profile_length_m}, side={side}",
        )
        logger.debug(f"Profile creation error: {exc}")
        return gpd.GeoDataFrame(geometry=[], crs=25833), None

    masked_dem = _mask_dem_array(dem_array, mask_array)

    retro_distance_list: list[float | None] = []
    max_heights_list: list[float] = []
    distances_list: list[Any] = []
    elevations_list: list[Any] = []
    slope_line_plot_list: list[Any] = []

    for profile_line in profiles:
        distances, elevations = create_terrain_profile(
            line=profile_line,
            dem_array=masked_dem,
            profile=dem_profile,
            resolution=resolution_m,
        )
        distances_list.append(distances)
        elevations_list.append(elevations)

        best_slope_line, _ = find_first_valid_retrogression_line(
            distances,
            elevations,
            depth=calc_depth,
            slope_ratio=slope_ratio,
            max_distance=tolerance_distance_m,
            min_height=min_height_m,
            debug=False,
        )

        if best_slope_line is not None:
            vertical_distances = np.array(elevations) - best_slope_line
            slope_line_plot = best_slope_line
        else:
            vertical_distances = depth_above_slope(distances, elevations, depth=calc_depth, slope_ratio=slope_ratio)
            slope_line_plot = [elevations[0] - calc_depth + slope_ratio * distance for distance in distances]

        slope_line_plot_list.append(slope_line_plot)

        current_retro_distance = retrogression_distance(distances, vertical_distances, ignore_first_meters=0)
        if current_retro_distance is not None:
            mask_dist = np.array(distances) <= current_retro_distance
            valid_diffs = np.array(elevations)[mask_dist] - (elevations[0] - calc_depth)
            valid_diffs = valid_diffs[~np.isnan(valid_diffs)]
            max_height = float(np.max(valid_diffs)) if len(valid_diffs) > 0 else 0
        else:
            max_height = 0

        retro_distance_list.append(current_retro_distance if max_height_m >= max_height >= min_height_m else 0)
        max_heights_list.append(max_height)

    if all(distance < resolution_m for distance in retro_distance_list if distance is not None):
        if debug:
            logger.debug("No release to plot")
        return gpd.GeoDataFrame(geometry=[], crs=25833), None

    coordinates = get_coordinates_for_retrogression_distance(profiles, retro_distance_list)
    points_coords = [Point(coordinate) for coordinate in coordinates if coordinate is not None]
    if len(points_coords) < min_n_points_for_envelope:
        return gpd.GeoDataFrame(geometry=[], crs=25833), None

    line_coords = _line_vertex_points(line)
    envelope = generate_envelope_around_points(points_coords + line_coords)
    release_gdf = gpd.GeoDataFrame(
        geometry=[envelope],
        data={"slope": slope_ratio, "max_height": max(max_heights_list)},
        index=[0],
        crs=25833,
    )

    if debug:
        from .profile_visualization import plot_release_from_profiles as _plot_release_from_profiles

        _plot_release_from_profiles(
            line,
            profiles,
            slope_ratio,
            distances_list,
            elevations_list,
            max_heights_list,
            retro_distance_list,
            slope_line_plot_list,
            points_coords,
            release_gdf,
        )

    debug_layers: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame] = (
        gpd.GeoDataFrame(geometry=profiles, crs=25833),
        gpd.GeoDataFrame(geometry=line_coords, crs=25833),
        gpd.GeoDataFrame(geometry=points_coords, crs=25833),
        gpd.GeoDataFrame(geometry=[line], crs=25833),
    )
    return release_gdf, debug_layers


def find_first_valid_retrogression_line(
    distances: list[float] | np.ndarray,
    elevations: list[float] | np.ndarray,
    depth: float,
    slope_ratio: float,
    max_distance: float,
    min_height: float = 0,
    debug: bool = False,
) -> tuple[np.ndarray | None, float | None]:
    """Find the first valid retrogression line that stays below terrain."""

    dist_arr = np.array(distances)
    elev_arr = np.array(elevations)
    if len(dist_arr) == 0:
        return None, None

    delta_dist = dist_arr[None, :] - dist_arr[:, None]
    lines_matrix = (elev_arr[0] - depth) + (delta_dist * slope_ratio)
    lines_matrix[np.tril_indices(lines_matrix.shape[0], k=-1)] = np.nan

    if debug:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(distances, elevations, color="black", linewidth=2, label="Terrain")
        step = max(1, int(len(distances) / 20))
        plt.plot(distances, lines_matrix.T[:, ::step], color="red", alpha=0.3, linewidth=0.5)
        plt.title(f"Matrix of parallel lines (Slope: {slope_ratio:.3f})")
        plt.xlabel("Distance")
        plt.ylabel("Elevation")
        plt.legend(["Terrain", "Candidate Lines"])
        plt.show()

    indices_within_max = np.where(dist_arr <= max_distance)[0]
    if indices_within_max.size == 0:
        diffs = elev_arr - lines_matrix[0, :]
        if np.nanmax(diffs) >= min_height:
            return np.minimum(lines_matrix[0, :], elev_arr), distances[0]
        return None, None

    idx_limit_check = indices_within_max[-1]
    depth_matrix = elev_arr - lines_matrix

    valid_idx: int | None = None
    current_max_height = 0.0

    for index in indices_within_max:
        start_check = index + 1
        end_check = idx_limit_check + 1

        if start_check >= len(dist_arr):
            continue

        if start_check >= end_check:
            segment_is_valid = True
        else:
            segment_is_valid = bool(np.all(depth_matrix[index, start_check:end_check] > 0))

        if not segment_is_valid:
            continue

        heights_above_base = elev_arr[index:] - elev_arr[0]
        if len(heights_above_base) == 0:
            continue

        current_max_height = float(np.max(heights_above_base))
        if current_max_height >= min_height:
            valid_idx = index
            break

    if valid_idx is None:
        corrected_slope_line = None
        new_start_distance = None
    else:
        corrected_slope_line = lines_matrix[valid_idx, :].copy()
        for idx in np.where(dist_arr <= max_distance)[0]:
            if (
                idx >= valid_idx
                and not np.isnan(corrected_slope_line[idx])
                and corrected_slope_line[idx] > elev_arr[idx]
            ):
                corrected_slope_line[idx] = elev_arr[idx] - 0.01
        new_start_distance = distances[valid_idx]

    if debug:
        status = "Found valid line" if valid_idx is not None else "No valid line found"
        logger.debug(status)
        if valid_idx is not None and new_start_distance is not None:
            logger.debug(f"Index: {valid_idx} ({new_start_distance:.2f} m), Max Height: {current_max_height:.2f}m")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(distances, elevations, color="black", linewidth=2, label="Terrain")
        if corrected_slope_line is not None and valid_idx is not None:
            plt.plot(
                distances,
                lines_matrix[valid_idx, :],
                color="red",
                alpha=0.5,
                linestyle="--",
                label="Original Line",
            )
            plt.plot(
                distances,
                corrected_slope_line,
                color="blue",
                linewidth=2,
                label="Draped Line",
            )
        plt.axvline(x=max_distance, color="gray", linestyle="--", label="Max Distance Check")
        plt.axhline(y=elev_arr[0], color="green", linestyle=":", alpha=0.5, label="Base Level")
        plt.title(f"Result: {status}")
        plt.legend()
        plt.show()

    return corrected_slope_line, new_start_distance


def compute_slope_line_terrain_intersection(
    distances: np.ndarray,
    elevations: np.ndarray,
    initial_depth: float,
    slope_ratio: float = 0.25,
    max_distance: float = 100.0,
    start_tolerance: float = 20.0,
) -> tuple[float, float, bool, float]:
    """Compute the best slope-line/terrain intersection within a tolerance window."""

    start_elevation = elevations[0] - initial_depth
    start_indices = np.where(distances <= start_tolerance)[0]

    best_height_diff = 0.0
    best_exit_dist = 0.0
    best_intersects = False
    best_start_offset = 0.0

    for start_idx in start_indices:
        start_offset = distances[start_idx]
        adjusted_distances = distances[start_idx:] - start_offset
        adjusted_elevations = elevations[start_idx:]
        if len(adjusted_elevations) == 0:
            continue

        line_elevations = start_elevation + slope_ratio * adjusted_distances
        below_terrain = line_elevations < adjusted_elevations
        within_distance = adjusted_distances <= max_distance
        intersection_zone = below_terrain & within_distance
        if not np.any(intersection_zone):
            continue

        current_height_diff = 0.0
        current_exit_dist = 0.0
        current_intersects = False

        for index in range(1, len(adjusted_distances)):
            if not within_distance[index]:
                break
            if below_terrain[index - 1] and not below_terrain[index]:
                current_height_diff = adjusted_elevations[index] - start_elevation
                current_exit_dist = adjusted_distances[index] + start_offset
                current_intersects = True
                break

        if not current_intersects and np.any(within_distance):
            idx = np.where(within_distance)[0][-1]
            if below_terrain[idx]:
                current_height_diff = adjusted_elevations[idx] - start_elevation
                current_exit_dist = adjusted_distances[idx] + start_offset
                current_intersects = True

        if current_intersects and current_height_diff > best_height_diff:
            best_height_diff = current_height_diff
            best_exit_dist = current_exit_dist
            best_intersects = True
            best_start_offset = start_offset

    return best_height_diff, best_exit_dist, best_intersects, best_start_offset


def plot_terrain_profile_with_slope_line(
    distances: np.ndarray,
    elevations: np.ndarray,
    initial_depth: float,
    slope_ratio: float = 0.25,
    max_distance: float = 100.0,
    start_tolerance: float = 20.0,
    figsize: tuple[int, int] = (12, 6),
    title: str = "Terrain Profile with Slope Line Analysis",
) -> None:
    """Plot a terrain profile with the best-fitting slope line."""

    from .profile_visualization import plot_terrain_profile_with_slope_line as _plot_terrain_profile_with_slope_line

    _plot_terrain_profile_with_slope_line(
        distances=distances,
        elevations=elevations,
        initial_depth=initial_depth,
        slope_ratio=slope_ratio,
        max_distance=max_distance,
        start_tolerance=start_tolerance,
        figsize=figsize,
        title=title,
    )


def plot_release_from_profiles(
    line: Any,
    profiles: list[Any],
    slope_ratio: float,
    distances_list: list[Any],
    elevations_list: list[Any],
    max_heights_list: list[float],
    retro_distance_list: list[float | None],
    slope_line_plot_list: list[Any],
    points_coords: list[Point],
    return_gdf: gpd.GeoDataFrame,
) -> None:
    """Plot profile diagnostics and the resulting release envelope."""

    from .profile_visualization import plot_release_from_profiles as _plot_release_from_profiles

    _plot_release_from_profiles(
        line=line,
        profiles=profiles,
        slope_ratio=slope_ratio,
        distances_list=distances_list,
        elevations_list=elevations_list,
        max_heights_list=max_heights_list,
        retro_distance_list=retro_distance_list,
        slope_line_plot_list=slope_line_plot_list,
        points_coords=points_coords,
        return_gdf=return_gdf,
    )
