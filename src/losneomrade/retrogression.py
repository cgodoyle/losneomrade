from __future__ import annotations

import logging
import warnings
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry.base import BaseGeometry

from . import utils

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def _get_tqdm() -> Any:
    """Import tqdm lazily so worker processes avoid the UI stack at import time."""

    from tqdm.auto import tqdm

    return tqdm


def run_retrogression(
    bounds: tuple | None,
    rel_shape: gpd.GeoDataFrame,
    point_depth: float = 0.0,
    mask: gpd.GeoDataFrame | None = None,
    min_slope: float = 1 / 15,
    min_height: float = 5,
    min_length: float = 75,
    custom_raster: str | None = None,
    return_animation: bool = False,
    verbose: bool = True,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[np.ndarray]]:
    """Run landslide retrogression from a release area.

    Args:
        bounds: Bounding box as (xmin, ymin, xmax, ymax). None if custom_raster is used.
        rel_shape: Release area as a GeoDataFrame (any geometry type).
        point_depth: Depth of the source points/line/polygon in meters.
        mask: Optional clipping mask as GeoDataFrame (e.g. from masks.get_msml_mask).
        min_slope: Minimum slope of the failure line. Default 1/15 per NVE guidelines.
        min_height: Minimum height for slope criterion in meters.
        min_length: Minimum length before slope is checked in meters.
        custom_raster: Path to custom raster file (tif) for calculations.
        return_animation: Whether to return the animation frames.
        verbose: Whether to log progress.

    Returns:
        GeoDataFrame with propagated release area. If return_animation is True,
        returns a tuple of (GeoDataFrame, list of animation frames).
    """
    if custom_raster is None:
        assert bounds is not None, "bounds required when custom_raster is not provided"
        dem_data = utils.get_hoydedata(bounds)
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]

    if mask is not None:
        mask_msml = utils.rasterize_shape(mask, dem_profile)
    else:
        mask_msml = None

    rel = utils.rasterize_shape(rel_shape, dem_profile)

    release, anim = landslide_retrogression(
        dem_array,
        rel,
        dem_profile["transform"],
        initial_release_depth=point_depth,
        min_slope=min_slope,
        min_height=min_height,
        min_length=min_length,
        mask=mask_msml,
        verbose=verbose,
    )

    akt = utils.polygonize_results(release, dem_profile, field="slope").to_crs(epsg=25833)
    if return_animation:
        akt = (akt, anim)
    return akt


def landslide_retrogression(
    dem: np.ndarray,
    initial_release: np.ndarray,
    dem_transform: rasterio.transform.Affine,
    min_slope: float = 1 / 15,
    min_height: float = 5,
    min_length: float = 200,
    max_length: float = 2000,
    initial_release_depth: float = 0,
    mask: np.ndarray | None = None,
    verbose: bool = False,
    slope_chunk_size: int = 1000,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Propagate a landslide from a release area in a DEM using BFS.

    Stop criteria are defined by maximum slope, minimum and maximum length.
    Propagation is done iteratively from the release area outward in 3D.

    Phase 1 (Unconditional): Expands release area up to min_length.
    Phase 2 (Conditional BFS): Uses a front of candidate pixels, checking
    slope criteria only for candidates rather than the full release area.

    Args:
        dem: DEM elevation array.
        initial_release: Initial release area as a boolean array (same shape as DEM).
        dem_transform: Affine transformation of the DEM/release.
        min_slope: Minimum slope of the failure line. Default 1/15 per NVE guidelines.
        min_height: Minimum height for slope criterion in meters.
        min_length: Minimum propagation length in meters.
        max_length: Maximum propagation length in meters.
        initial_release_depth: Depth of the initial release area in meters.
        mask: Binary mask for analysis area (same shape as DEM). None means no mask.
        verbose: Whether to log progress.
        slope_chunk_size: Chunk size for slope calculation to manage memory.

    Returns:
        Tuple of (release array, list of animation frames).
    """
    if verbose:
        logger.info("Running landslide propagation (Optimized BFS)...")
    if abs(round(dem_transform[0], 2)) != abs(round(dem_transform[4], 2)):
        logger.warning("DEM is not square")

    res = abs(dem_transform[0])

    min_iter = int(min_length // res)
    max_iter = int(max_length // res)

    # shut up RuntimeWarning
    np.seterr(divide="ignore", invalid="ignore")

    initial_release = initial_release.astype(bool)

    # 1. Setup Source Points (Constant)
    i_rel, j_rel = np.where(initial_release == 1)
    x_rel, y_rel = rasterio.transform.xy(dem_transform, i_rel, j_rel)
    z_rel = np.array([dem[ii, jj] - initial_release_depth for ii, jj in zip(i_rel, j_rel)])
    source_coords = np.c_[x_rel, y_rel, z_rel]

    animation = [initial_release]

    # 2. Phase 1: Unconditional Expansion (min_length)
    current_release = initial_release.copy()

    # We iterate to generate animation frames and handle masking properly step-by-step
    # (though we could optimize this if animation is not needed, but let's keep it safe)
    for i in range(min_iter):
        # Dilate by 1 to get the rim
        buffered = create_buffer(current_release, 1)
        # Apply mask
        buffered = apply_mask(buffered, mask)

        if not np.any(buffered):
            break

        # Add to release
        current_release = current_release | buffered
        animation.append(current_release.copy())

    release = current_release

    # 3. Phase 2: Conditional Expansion (BFS)
    # Checked mask: pixels we have already processed (either accepted or rejected)
    # Initially, everything in the current release is "checked" (accepted).
    checked = release.copy()

    # Current candidates: neighbors of the current release that are NOT checked
    # create_buffer returns the rim.
    candidates_mask = create_buffer(release, 1)
    candidates_mask = apply_mask(candidates_mask, mask)
    candidates_mask = candidates_mask & (~checked)

    n_iter = min_iter

    tqdm = _get_tqdm()

    with tqdm(total=max_iter, initial=n_iter, desc="iterations", disable=not verbose) as pbar:
        while n_iter < max_iter:
            if not np.any(candidates_mask):
                break

            # Extract candidate coordinates
            i_cand, j_cand = np.where(candidates_mask == 1)
            x_cand, y_cand = rasterio.transform.xy(dem_transform, i_cand, j_cand)
            z_cand = np.array([dem[ii, jj] for ii, jj in zip(i_cand, j_cand)])
            cand_coords = np.c_[x_cand, y_cand, z_cand]

            # Optimization: Filter source points to relevant area
            # We only care about source points that could possibly satisfy the slope condition.
            # Max relevant distance is bounded by max_length (since we stop there)
            # or by the physical limit (delta_z / min_slope).
            # We use a generous buffer to be safe.
            search_buffer = max(max_length, 2000)

            c_xmin, c_ymin = np.min(cand_coords[:, :2], axis=0)
            c_xmax, c_ymax = np.max(cand_coords[:, :2], axis=0)

            s_xmin, s_ymin = c_xmin - search_buffer, c_ymin - search_buffer
            s_xmax, s_ymax = c_xmax + search_buffer, c_ymax + search_buffer

            # Filter source points (vectorized)
            relevant_mask = (
                (source_coords[:, 0] >= s_xmin)
                & (source_coords[:, 0] <= s_xmax)
                & (source_coords[:, 1] >= s_ymin)
                & (source_coords[:, 1] <= s_ymax)
            )

            relevant_sources = source_coords[relevant_mask]

            if len(relevant_sources) == 0:
                slopes = np.zeros(len(cand_coords))
            else:
                # Compute slopes against filtered source
                if slope_chunk_size is not None:
                    slopes = utils.compute_slope_chunked(
                        cand_coords,
                        relevant_sources,
                        h_min=min_height,
                        chunk_size=slope_chunk_size,
                    )
                else:
                    slopes = utils.compute_slope(cand_coords, relevant_sources, h_min=min_height)

            # Identify successful candidates
            success_mask_local = slopes > min_slope

            # If no success, this front stops.
            if not np.any(success_mask_local):
                # Mark all as checked (rejected)
                checked[i_cand, j_cand] = 1
                break

            # Update release with successful candidates
            # We need to map back to global grid
            i_success = i_cand[success_mask_local]
            j_success = j_cand[success_mask_local]

            new_release_pixels = np.zeros_like(release, dtype=bool)
            new_release_pixels[i_success, j_success] = 1

            release = release | new_release_pixels

            # Mark ALL current candidates as checked (both success and fail)
            checked[i_cand, j_cand] = 1

            animation.append(release.copy())

            # Generate NEXT candidates
            # Only neighbors of the NEWLY ADDED pixels need to be checked.
            new_candidates = create_buffer(new_release_pixels, 1)
            new_candidates = apply_mask(new_candidates, mask)

            # Filter out already checked
            candidates_mask = new_candidates & (~checked)

            n_iter += 1
            pbar.update(1)

    if np.all(release == initial_release) and min_iter > 0:
        # This handles the case where min_iter > 0 but masking prevented any expansion
        # Or if min_iter=0 and no propagation happened.
        pass

    return release, animation


def run_retrogression_with_initial_landslide(
    bounds: tuple | None,
    rel_shape: list[BaseGeometry],
    point_depth: float = 0.0,
    mask: gpd.GeoDataFrame | None = None,
    ini_slope: float = 1 / 4,
    retro_slope: list[float] | float = 1 / 15,
    min_height: float = 5,
    min_length: float = 75,
    custom_raster: str | None = None,
    return_animation: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, list[np.ndarray]]:
    """Run landslide retrogression with an initial landslide phase.

    First propagates using ini_slope (steeper, initial failure), then
    continues with retro_slope (gentler, retrogressive phase).

    Args:
        bounds: Bounding box as (xmin, ymin, xmax, ymax). None if custom_raster is used.
        rel_shape: Release area as a list of shapely geometries.
        point_depth: Depth of the source points/line/polygon in meters.
        mask: Optional clipping mask as GeoDataFrame (e.g. from masks.get_msml_mask).
        ini_slope: Slope for the initial landslide phase.
        retro_slope: Retrogressive slope(s) for second phase.
        min_height: Minimum height for slope criterion in meters.
        min_length: Minimum length before slope is checked in meters.
        custom_raster: Path to custom raster file (tif) for calculations.
        return_animation: Whether to return the animation frames.

    Returns:
        GeoDataFrame with propagated release area. If return_animation is True,
        returns a tuple of (GeoDataFrame, list of animation frames).
    """
    if not isinstance(retro_slope, list):
        retro_slope = [retro_slope]

    if custom_raster is None:
        assert bounds is not None, "bounds required when custom_raster is not provided"
        dem_data = utils.get_hoydedata(bounds)
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]

    if mask is not None:
        mask_msml = utils.rasterize_shape(mask, dem_profile)
    else:
        mask_msml = None

    rel = utils.rasterize_shape(rel_shape, dem_profile)

    min_length_first = min_length * ini_slope
    min_length_second = min_length - min_length_first

    release_first, animation_first = landslide_retrogression(
        dem_array,
        rel,
        dem_profile["transform"],
        initial_release_depth=point_depth,
        min_slope=ini_slope,
        min_height=min_height,
        min_length=min_length_first,
        mask=mask_msml,
        verbose=False,
    )

    if np.all(release_first == rel) or release_first.sum() == 0:
        akt = gpd.GeoDataFrame(columns=["geometry", "slope"], crs=25833)
        animation_second = []

    else:
        first_release = utils.polygonize_results(release_first, dem_profile, field="slope").to_crs(epsg=25833)
        first_release["slope"] = ini_slope

        release_list = [first_release]

        for slope in retro_slope:
            release_second, animation_second = landslide_retrogression(
                dem=dem_array,
                initial_release=release_first,
                dem_transform=dem_profile["transform"],
                min_slope=slope,
                min_height=0,
                min_length=min_length_second,
                max_length=2000,
                initial_release_depth=0,
                mask=mask_msml,
                verbose=False,
            )

            second_release = utils.polygonize_results(release_second, dem_profile, field="slope").to_crs(epsg=25833)
            second_release["slope"] = slope
            release_list.append(second_release)

        akt = gpd.GeoDataFrame(pd.concat(release_list, ignore_index=True))

    if return_animation:
        animation = animation_first + [animation_first[-1] for _ in range(100)] + animation_second
        akt = (akt, animation)
    return akt


def apply_mask(array: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Apply a binary mask to a numpy array.

    Args:
        array: Input array.
        mask: Binary mask array, or None to skip masking.

    Returns:
        Masked copy of the array (zeros where mask is 0).
    """
    if mask is None:
        return array
    if array.shape != mask.shape:
        raise ValueError("Array and mask must have the same shape.")
    masked_array = array.copy()
    masked_array[mask == 0] = 0
    return masked_array


def landslide_retrogression_legacy(
    dem: np.ndarray,
    initial_release: np.ndarray,
    dem_transform: rasterio.transform.Affine,
    min_slope: float = 1 / 15,
    min_height: float = 5,
    min_length: float = 200,
    max_length: float = 2000,
    initial_release_depth: float = 0,
    mask: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Propagate a landslide from a release area (legacy non-BFS implementation).

    Args:
        dem: DEM elevation array.
        initial_release: Initial release area as a boolean array (same shape as DEM).
        dem_transform: Affine transformation of the DEM/release.
        min_slope: Minimum slope of the failure line. Default 1/15 per NVE guidelines.
        min_height: Minimum height for slope criterion in meters.
        min_length: Minimum propagation length in meters.
        max_length: Maximum propagation length in meters.
        initial_release_depth: Depth of the initial release area in meters.
        mask: Binary mask for analysis area (same shape as DEM). None means no mask.
        verbose: Whether to log progress.

    Returns:
        Tuple of (release array, list of animation frames).
    """
    if verbose:
        logger.info("Running landslide propagation (legacy)...")
    if abs(round(dem_transform[0], 2)) != abs(round(dem_transform[4], 2)):
        logger.warning("DEM is not square")

    res = abs(dem_transform[0])

    min_iter = int(min_length // res)
    max_iter = int(max_length // res)

    # shut up RuntimeWarning
    np.seterr(divide="ignore", invalid="ignore")

    n_iter = 1

    release = initial_release.copy()

    i_rel, j_rel = np.where(initial_release == 1)
    x_rel, y_rel = rasterio.transform.xy(dem_transform, i_rel, j_rel)
    z_rel = np.array([dem[ii, jj] - initial_release_depth for ii, jj in zip(i_rel, j_rel)])
    release_coords = np.c_[x_rel, y_rel, z_rel]

    animation = []
    animation.append(initial_release)

    initial_release_buffered = apply_mask(initial_release + create_buffer(initial_release, min_iter), mask)

    tqdm = _get_tqdm()

    with tqdm(total=0, desc="iterations", disable=not verbose) as pbar:
        while n_iter < max_iter:
            buffered = apply_mask(create_buffer(release, 1), mask)

            i_buffered, j_buffered = np.where(buffered == 1)
            x_buffered, y_buffered = rasterio.transform.xy(dem_transform, i_buffered, j_buffered)
            z_buffered = np.array([dem[ii, jj] for ii, jj in zip(i_buffered, j_buffered)])
            buffered_coords = np.c_[x_buffered, y_buffered, z_buffered]

            # h_min = 0 if n_iter <= min_iter else min_height
            slopes = utils.compute_slope(buffered_coords, release_coords, h_min=min_height)

            if n_iter > min_iter:
                neighbours_filtered = [
                    (i_buffered[ii], j_buffered[ii]) for ii in list(np.where(np.array(slopes) > min_slope)[0])
                ]

                release_after = release.copy()

                for ii in neighbours_filtered:
                    release_after[ii] = 1
            else:
                release_after = release + buffered

            release_after = apply_mask(release_after, mask)

            if np.all(release.astype(bool) == release_after.astype(bool)) and n_iter > min_iter:
                break

            release = release_after.copy()

            animation.append(release_after)

            n_iter += 1
            pbar.update(1)

    if np.all(release == initial_release_buffered):
        logger.warning(
            f"No propagation beyond minimum length of {min_length} m / {min_iter + 1} iterations. "
            "Returning original release area.",
        )
        release = initial_release
    return release, animation


def create_buffer(image: np.ndarray, buffer_size: int = 1) -> np.ndarray:
    """Create a buffer ring around a binary image via binary dilation.

    Args:
        image: Binary image array.
        buffer_size: Size of the buffer in pixels.

    Returns:
        Buffer ring as a boolean array (dilated minus original).
    """
    from scipy.ndimage import binary_dilation

    dilated_image = binary_dilation(image, iterations=buffer_size)
    # buffer = ((dilated_image - image) > 0).astype(bool)
    buffer = dilated_image & (~image.astype(bool))

    return buffer


def animate_landslide_retrogresion(
    animation: list[np.ndarray],
    dem: np.ndarray,
    frame_step: int | None = None,
) -> object:
    """Create a plotly animation of the landslide retrogression."""

    from .retrogression_visualization import animate_landslide_retrogresion as _animate_landslide_retrogresion

    return _animate_landslide_retrogresion(animation=animation, dem=dem, frame_step=frame_step)


def hillshade_img(dem_array: np.ndarray, ve: float = 1) -> object:
    """Create a plotly Image object of a hillshade."""

    from .retrogression_visualization import hillshade_img as _hillshade_img

    return _hillshade_img(dem_array=dem_array, ve=ve)


def plot_hillshade_overlay(
    dem: np.ndarray,
    overlay: np.ndarray,
    ve: float = 1,
    alpha: float = 0.4,
    res: float = 5,
    figsize: tuple[float, float] = (10, 10),
) -> Any:
    """Plot a hillshade overlaid with a binary overlay."""

    from .retrogression_visualization import plot_hillshade_overlay as _plot_hillshade_overlay

    return _plot_hillshade_overlay(
        dem=dem,
        overlay=overlay,
        ve=ve,
        alpha=alpha,
        res=res,
        figsize=figsize,
    )


def gen_animation(
    dem: np.ndarray,
    animation: list[np.ndarray],
    skip_frames: int = 10,
    filename: str | None = None,
) -> list:
    """Generate a GIF animation from retrogression frames."""

    from .retrogression_visualization import gen_animation as _gen_animation

    return _gen_animation(dem=dem, animation=animation, skip_frames=skip_frames, filename=filename)


def save_frames(dem_array: np.ndarray, animation: list[np.ndarray], out_dir: str, skip_frames: int = 10) -> None:
    """Save retrogression animation frames as PNG images."""

    from .retrogression_visualization import save_frames as _save_frames

    _save_frames(dem_array=dem_array, animation=animation, out_dir=out_dir, skip_frames=skip_frames)
