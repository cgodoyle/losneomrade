import io
import os
import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image
from scipy.ndimage import binary_dilation, label, find_objects
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial

from . import utils

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_retrogression(bounds: tuple,
                      rel_shape: gpd.GeoDataFrame,
                      point_depth: float = 0.0,
                      clip_to_msml=False,
                      min_slope: float = 1 / 15,
                      min_height: float = 5,
                      min_length: float = 75,
                      slope_chunk_size: int = 1000,
                      custom_raster=None,
                      return_animation=False,
                      verbose=True) -> gpd.GeoDataFrame:
    """
    Wrapper function to run landslide retrogression (in a similar way to terrain_criteria.terrain_criteria). 

    Args:

        bounds (tuple): xmin,xmax,ymin,ymax of the calculation window
        rel_shape (gpd.GeoDataFrame): release area as a geodataframe (any type of geometry)
        point_depth (float): depth of the source points (/line/polygon)
        clip_to_msml (bool): wheter to clip against MSML (sammenhengede forekomster).
        min_slope (float): minimum slope of the landslide/slope of the failure line. 
                            Default is 1/15 as in NVE's guidelines
        min_height (float): minimum height for checking the slope criterion. Default is 5 m.
        min_length (float): minimum length of the landslide (slope not checked within this length). Default is 75 m.
        slope_chunk_size (int): chunk size for slope calculation to lower memory/CPU peaks without
                                changing results. Uses utils.compute_slope_chunked. Default is 1000.
        custom_raster (np.ndarray): custom raster to use for the calculation. Default is None.
        return_animation (bool): wheter to return the animation of the retrogression. Default is False.
        verbose (bool): wheter to print progress. Default is True.

    Returns:
        akt (gpd.GeoDataFrame): propagated release area of the landslide as a geodataframe
        animation (list): list of numpy arrays with the landslide retrogression. Only returned if return_animation=True.

    """
    if custom_raster is None:
        dem_data = utils.get_hoydedata(bounds, )
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]

    if clip_to_msml:
        mask_gpd = utils.get_msml_mask((bounds[0], bounds[2], bounds[1], bounds[3]))
        mask_msml = utils.rasterize_shape(mask_gpd, dem_profile)
    else:
        mask_msml = None

    rel = utils.rasterize_shape(rel_shape, dem_profile)

    release, anim = landslide_retrogression_optimized(
        dem_array, rel, dem_profile["transform"], initial_release_depth=point_depth,
        min_slope=min_slope, min_height=min_height, min_length=min_length, mask=mask_msml,
        verbose=verbose, slope_chunk_size=slope_chunk_size)

    akt = utils.polygonize_results(release, dem_profile, field="slope").to_crs(epsg=25833)
    if return_animation:
        akt = (akt, anim)
    return akt



def run_retrogression_with_initial_landslide(
        bounds: tuple,
        rel_shape: gpd.GeoDataFrame,
        point_depth: float = 0.0,
        clip_to_msml=False,
        custom_msml: gpd.GeoDataFrame=None,
        ini_slope: float = 1 / 4,
        retro_slope: list = [1 / 15],
        min_height: float = 5,
        min_length: float = 75,
        slope_chunk_size: int = 1000,
        custom_raster=None,
        return_animation=False,

):
    """
    Run landslide retrogression with an initial landslide.

    Args:
        bounds (tuple): xmin, ymin, xmax, ymax
        rel_shape (gpd.GeoDataFrame): release area as a geodataframe (any type of geometry)
        point_depth (float): depth of the source points (/line/polygon)
        clip_to_msml (bool): wheter to clip against MSML (sammenhengede forekomster).
        ini_slope (list): list with slope of the landslide's release area to compute.
        retro_slope (float): retrogressive slope of the landslide.
        min_height (float): minimum height for checking the slope criterion. Default is 5 m.
        min_length (float): minimum length of the landslide (slope not checked within this length). Default is 75 m.
        slope_chunk_size (int): chunk size for slope calculation to lower memory/CPU peaks without
                                changing results. Uses utils.compute_slope_chunked. Default is 1000.
        custom_raster (np.ndarray): custom raster to use for the calculation. Default is None.
        return_animation (bool): wheter to return the animation of the retrogression. Default is False.

    """
    if not isinstance(retro_slope, list):
        retro_slope = [retro_slope]
        
    if custom_raster is None:
        dem_data = utils.get_hoydedata(bounds, )
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]

    if clip_to_msml:
        mask_gpd = utils.get_msml_mask((bounds[0], bounds[2], bounds[1], bounds[3]))
        mask_msml = utils.rasterize_shape(mask_gpd, dem_profile)

    else:
        if custom_msml is not None:
            mask_msml = utils.rasterize_shape(custom_msml, dem_profile)
        else:
            mask_msml = None

    rel = utils.rasterize_shape(rel_shape, dem_profile)

    min_length_first = min_length * ini_slope
    min_length_second = min_length - min_length_first

    
    release_first, animation_first = landslide_retrogression_optimized(
        dem_array,
        rel, dem_profile["transform"],
        initial_release_depth=point_depth,
        min_slope=ini_slope,
        min_height=min_height,
        min_length=min_length_first,
        mask=mask_msml,
        verbose=False,
        slope_chunk_size=slope_chunk_size)

    if np.all(release_first == rel) or release_first.sum() == 0:
        
        akt = gpd.GeoDataFrame(columns=["geometry", "slope"], crs=25833)
        animation_second = []

    else:

        first_release = utils.polygonize_results(release_first, dem_profile, field="slope").to_crs(epsg=25833)
        first_release["slope"] = ini_slope

        release_list = [first_release]

        for slope in retro_slope:

            release_second, animation_second = landslide_retrogression_optimized(
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
                slope_chunk_size=slope_chunk_size
            )
            

            second_release = utils.polygonize_results(release_second, dem_profile, field="slope").to_crs(epsg=25833)
            second_release["slope"] = slope
            release_list.append(second_release)

        akt = pd.concat(release_list, ignore_index=True)


    if return_animation:
        animation = animation_first + [animation_first[-1] for _ in range(100)] + animation_second
        akt = (akt, animation)
    return akt

  
def apply_mask(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to a numpy array.

    Args:
        array (np.ndarray): Input array.
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Masked array.
    """
    if mask is None:
        return array
    if array.shape != mask.shape:
        raise ValueError("Array and mask must have the same shape.")
    masked_array = array.copy()
    masked_array[mask == 0] = 0
    return masked_array

def landslide_retrogression_optimized(dem: np.ndarray,
                            initial_release: np.ndarray,
                            dem_transform: rasterio.transform.Affine,
                            min_slope: float = 1 / 15,
                            min_height: float = 5,
                            min_length: float = 200,
                            max_length: float = 2000,
                            initial_release_depth: float = 0,
                            mask: np.ndarray = None,
                            verbose: bool = False,
                            slope_chunk_size: int = 1000):
    """
    Propagates a landslide from a release area in a DEM. Stop criteria is defined by the maximum slope, minimum and
    maximum length of the landslide. The propagation is done iteratively, starting from the release area and moving
    outwards. The propagation is done in 3D, i.e. the landslide can propagate in any direction.

    **Optimization Changes (BFS):**
    This function has been optimized using a Breadth-First Search (BFS) approach for the conditional expansion phase.
    
    1.  **Phase 1 (Unconditional):** Expands the release area unconditionally up to `min_length`.
    2.  **Phase 2 (Conditional BFS):**
        -   Instead of checking every pixel in the release area at every iteration (which is O(N^2) or worse),
            we maintain a "front" of candidate pixels (neighbors of the current release).
        -   We only check slope criteria for these candidate pixels against relevant source points.
        -   Pixels that fail the criteria are marked as "checked" and not re-evaluated.
        -   Pixels that pass are added to the release, and their neighbors become new candidates.
        -   This reduces redundant calculations significantly.

    Parameters:
        dem (np.ndarray): DEM as a numpy array
        initial_release (np.ndarray): initial release area as a boolean numpy array.
                                      Must have the same shape and same transform as the DEM.
        dem_transform (Affine): affine transformation of the DEM/release.
        min_slope (float): minimum slope of the landslide. Default is 1/15 as in NVE's guidelines
        min_height (float): minimum height for checking the slope criterion. Default is 5 m.
        min_length (float): minimum length of the landslide. Default is 200 m.
        max_length (float): maximum length of the landslide. Default is 2000 m.
        initial_release_depth (float): depth of the initial release area. Default is 0.
        #TODO: change to depth in the raster (as pixel value) instead.
        mask (np.ndarray): mask of the area outside analysis. Must have the same shape and same transform as the DEM.
                            Default is None.
        verbose (bool): wheter to print progress. Default is False.
        slope_chunk_size (int): chunk size for slope calculation (default 1000). Uses
                                utils.compute_slope_chunked to keep memory and runtime in check while
                                preserving baseline results.


    Returns:
        release (np.ndarray): propagated release area of the landslide as a boolean numpy array


    """
    if verbose:
        print("Running landslide propagation (Optimized BFS)...")
    if abs(round(dem_transform[0], 2)) != abs(round(dem_transform[4], 2)):
        if verbose:
            print("Warning: DEM is not square")

    res = abs(dem_transform[0])

    min_iter = int(min_length // res)
    max_iter = int(max_length // res)

    # shut up RuntimeWarning
    np.seterr(divide='ignore', invalid='ignore')

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
                (source_coords[:, 0] >= s_xmin) & 
                (source_coords[:, 0] <= s_xmax) & 
                (source_coords[:, 1] >= s_ymin) & 
                (source_coords[:, 1] <= s_ymax)
            )
            
            relevant_sources = source_coords[relevant_mask]
            
            if len(relevant_sources) == 0:
                slopes = np.zeros(len(cand_coords))
            else:
                # Compute slopes against filtered source
                if slope_chunk_size is not None:
                    slopes = utils.compute_slope_chunked(
                        cand_coords, relevant_sources, h_min=min_height, chunk_size=slope_chunk_size
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


def landslide_retrogression_original(dem: np.ndarray,
                            initial_release: np.ndarray,
                            dem_transform: rasterio.transform.Affine,
                            min_slope: float = 1 / 15,
                            min_height: float = 5,
                            min_length: float = 200,
                            max_length: float = 2000,
                            initial_release_depth: float = 0,
                            mask: np.ndarray = None,
                            verbose: bool = False):
    """
    Propagates a landslide from a release area in a DEM. Stop criteria is defined by the maximum slope, minimum and
    maximum length of the landslide. The propagation is done iteratively, starting from the release area and moving
    outwards. The propagation is done in 3D, i.e. the landslide can propagate in any direction.

    Parameters:
        dem (np.ndarray): DEM as a numpy array
        initial_release (np.ndarray): initial release area as a boolean numpy array.
                                      Must have the same shape and same transform as the DEM.
        dem_transform (Affine): affine transformation of the DEM/release.
        min_slope (float): minimum slope of the landslide. Default is 1/15 as in NVE's guidelines
        min_height (float): minimum height for checking the slope criterion. Default is 5 m.
        min_length (float): minimum length of the landslide. Default is 200 m.
        max_length (float): maximum length of the landslide. Default is 2000 m.
        initial_release_depth (float): depth of the initial release area. Default is 0.
        #TODO: change to depth in the raster (as pixel value) instead.
        mask (np.ndarray): mask of the area outside analysis. Must have the same shape and same transform as the DEM.
                            Default is None.
        verbose (bool): wheter to print progress. Default is False.


    Returns:
        release (np.ndarray): propagated release area of the landslide as a boolean numpy array


    """
    if verbose:
        print("runing landslide propagation...")
    if abs(round(dem_transform[0], 2)) != abs(round(dem_transform[4], 2)):
        if verbose:
            print("Warning: DEM is not square")

    res = abs(dem_transform[0])

    min_iter = int(min_length // res)
    max_iter = int(max_length // res)

    # shut up RuntimeWarning
    np.seterr(divide='ignore', invalid='ignore')

    n_iter = 1

    release = initial_release.copy()

    i_rel, j_rel = np.where(initial_release == 1)
    x_rel, y_rel = rasterio.transform.xy(dem_transform, i_rel, j_rel)
    z_rel = np.array([dem[ii, jj] - initial_release_depth for ii, jj in zip(i_rel, j_rel)])
    release_coords = np.c_[x_rel, y_rel, z_rel]

    animation = []
    animation.append(initial_release)

    initial_release_buffered = apply_mask(initial_release + create_buffer(initial_release, min_iter), mask)

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
                neighbours_filtered = [(i_buffered[ii], j_buffered[ii]) for ii in
                                       list(np.where(np.array(slopes) > min_slope)[0])]

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
        if verbose:
            print(f"Warning: no propagation besides the minimum length of {min_length} m / {min_iter+1} iterations")
            print("returning the original release area")
        release = initial_release
    return release, animation


def create_buffer(image: np.ndarray, buffer_size: int = 1):
    """
    Create a buffer around an image by performing binary dilation.

    Args:
        image (np.ndarray): Image as a numpy array.
        buffer_size (int): Size of the buffer in pixels. Default is 1.

    Returns:
        np.ndarray: Buffer as a boolean numpy array.

    """
    dilated_image = binary_dilation(image, iterations=buffer_size)
    buffer = dilated_image & (~image.astype(bool))

    return buffer


def animate_landslide_retrogresion(animation: np.ndarray, dem: np.ndarray, frame_step: int = None) -> go.Figure:
    """
    Creates a plotly animation of the landslide retrogression.

    Args:

        animation (list): list of numpy arrays with the landslide retrogression.
        dem (np.ndarray): DEM as a numpy array.
        frame_step (int): step between frames. Default is len(animation)//5 if len(animation)//5 > 1 else 2.

    Returns:
        fig (plotly.graph_objects.Figure): figure object with the animation.

    """
    print("Animating landslide retrogression")

    if frame_step is None:
        frame_step = len(animation) // 5 if len(animation) // 5 > 1 else 2

    color_red = 'rgba(255, 0, 0, 0.5)'
    color_white = 'rgba(255, 255, 255, 0.0)'
    basemap = hillshade_img(dem, 1)
    fig_data = [basemap,
                go.Heatmap(z=animation[0], colorscale=[[0, color_white], [1, color_red]], showscale=False)]
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            title="Step 0",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None])])]
        ),
    )

    frames = [go.Frame(data=[basemap,
                             go.Heatmap(z=animation[i], colorscale=[[0, color_white], [1, color_red]],
                                        showscale=False)],
                       layout=go.Layout(title_text=f"Step {i}"))
              for i in range(1, len(animation), frame_step)]
    frames.append(go.Frame(data=[basemap,
                                 go.Heatmap(z=animation[-1], colorscale=[[0, color_white], [1, color_red]],
                                            showscale=False)],
                           layout=go.Layout(title_text=f"Step {len(animation)}")))
    fig.frames = frames

    height, width = dem.shape

    fig.update_xaxes(scaleanchor="y")
    fig.update_yaxes(scaleratio=1, autorange="reversed")
    fig.update_layout(xaxis_range=[0, width], yaxis_range=[0, height])
    fig.update_layout(width=500, height=500, coloraxis_showscale=False, plot_bgcolor=color_white,
                      )

    return fig


def hillshade_img(dem_array: np.ndarray, ve: int = 1) -> go.Image:
    """
    Create a plotly image object of a hillshade.

    Args:
        dem_array (np.ndarray): DEM as a numpy array.
        ve (float): vertical exaggeration of the hillshade. Default is 1.

    Returns:
        go.Image: plotly image object.

    """
    ls = LightSource(azdeg=315, altdeg=45)
    hilsh = ls.shade(dem_array, vert_exag=ve, blend_mode="hsv", cmap=plt.cm.gray, dx=5, dy=5)
    img = np.array((255 * hilsh[:, :, :3] + 0.5), int)
    return go.Image(z=img)


def plot_hillshade_overlay(dem: np.ndarray,
                           overlay: np.ndarray,
                           ve: int = 1,
                           alpha: float = 0.4,
                           res: float = 5,
                           figsize: tuple = (10, 10)) -> plt.figure:
    """
    Plot a hillshade overlayed with a binary overlay

    Args:
        dem (np.ndarray): DEM as a numpy array
        overlay (np.ndarray): overlay as a boolean numpy array.
        ve (float): vertical exaggeration of the hillshade. Default is 1.
        alpha (float): transparency of the overlay. Default is 0.4.
        res (float): resolution of the DEM. Default is 5.
        figsize (tuple): figure size. Default is (10,10).

    Returns:
        fig (matplotlib.figure.Figure): figure object

    """
    import matplotlib.colors as mcolors
    from matplotlib.colors import LightSource
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')

    cmap = mcolors.ListedColormap(['none', 'red'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ls = LightSource(azdeg=315, altdeg=45)

    fig, ax = plt.subplots(figsize=figsize)
    _ = ax.imshow(ls.hillshade(dem, vert_exag=ve, dx=res, dy=res), cmap='gray')
    _ = ax.imshow(overlay, cmap=cmap, norm=norm, alpha=alpha)

    plt.switch_backend(current_backend)

    return fig


def gen_animation(dem: np.ndarray, animation: list, skip_frames: int = 10, filename: str = None) -> list:
    """
    Generate a GIF animation from a list of matplotlib figures

    Args:
        dem (np.ndarray): DEM as a numpy array
        animation (list): list of numpy arrays
        skip_frames (int): number of frames to skip. Default is 10.
        filename (str): output filename. Default is None.

    Returns:
        frames (list): list of PIL images
    """
    fig_list = [plot_hillshade_overlay(dem, ani) for ani in animation[::skip_frames]]
    frames = []

    for fig in fig_list:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)

        img = Image.open(buffer)
        frames.append(img.copy())

        plt.clf()

    if filename is not None:
        frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=200)
    return frames


def save_frames(dem_array: np.ndarray, animation: list, out_dir: str, skip_frames: int = 10):
    """
    Save the frames of an animation as png images. Use with https://ezgif.com/ to generate the gif file.

    Args:
        dem_array (np.ndarray): DEM as a numpy array
        animation (list): list of numpy arrays
        out_dir (str): output directory
        skip_frames (int): number of frames to skip. Default is 10.

    Returns:
        None


    """
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')
    os.makedirs(out_dir, exist_ok=True)
    n_frames = len(animation[::skip_frames])
    for ii, ani in tqdm(enumerate(animation[::skip_frames]), total=n_frames, desc="saving frames"):
        fig = plot_hillshade_overlay(dem_array, ani)
        fig.savefig(f'{out_dir}\\gif_frame_{ii}.png')
        fig.clf()
    plt.switch_backend(current_backend)


# ============================================================================
# PARALLEL RETROGRESSION FUNCTIONS
# ============================================================================

def _process_component(args):
    """
    Helper function for parallel processing of retrogression components.
    """
    (dem_crop, rel_crop, transform_crop, depth, min_slope, min_height, min_length, mask_crop, slope_chunk_size) = args
    
    # Run retrogression on the crop
    result, _ = landslide_retrogression_optimized(
        dem_crop, rel_crop, transform_crop,
        initial_release_depth=depth,
        min_slope=min_slope,
        min_height=min_height,
        min_length=min_length,
        mask=mask_crop,
        verbose=False,
        slope_chunk_size=slope_chunk_size,
    )
    return result


def run_retrogression_parallel(bounds: tuple,
                               rel_shape: gpd.GeoDataFrame,
                               point_depth: float = 0.0,
                               clip_to_msml=False,
                               min_slope: float = 1 / 15,
                               min_height: float = 5,
                               min_length: float = 75,
                               slope_chunk_size: int = 1000,
                               custom_raster=None,
                               n_processes: int = None,
                               buffer_pixels: int = 50) -> gpd.GeoDataFrame:
    """
    Parallel version of run_retrogression.
    Splits the initial release area into connected components and processes them in parallel.
    NOTE: This version processes components independently and may produce different results
    than the baseline if components are close enough to interact during retrogression.
    Use run_retrogression_parallel_grouped for accurate results.
    slope_chunk_size is forwarded to landslide_retrogression to speed up slope checks without affecting outputs.
    """
    if custom_raster is None:
        dem_data = utils.get_hoydedata(bounds)
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]
    dem_transform = dem_profile["transform"]

    if clip_to_msml:
        mask_gpd = utils.get_msml_mask((bounds[0], bounds[2], bounds[1], bounds[3]))
        mask_msml = utils.rasterize_shape(mask_gpd, dem_profile)
    else:
        mask_msml = None

    # Rasterize release shape
    rel = utils.rasterize_shape(rel_shape, dem_profile)
    
    # Label connected components
    labeled_array, num_features = label(rel)
    
    if num_features == 0:
        return gpd.GeoDataFrame(columns=["geometry", "slope"], crs=25833)
        
    print(f"Processing {num_features} independent release zones in parallel...")
    
    # Prepare tasks with component index tracking
    tasks = []
    task_indices = []
    slices = find_objects(labeled_array)
    
    for i, sl in enumerate(slices):
        if sl is None: continue
        
        y_slice, x_slice = sl
        y_min = max(0, y_slice.start - buffer_pixels)
        y_max = min(dem_array.shape[0], y_slice.stop + buffer_pixels)
        x_min = max(0, x_slice.start - buffer_pixels)
        x_max = min(dem_array.shape[1], x_slice.stop + buffer_pixels)
        
        dem_crop = dem_array[y_min:y_max, x_min:x_max]
        rel_crop = (labeled_array[y_min:y_max, x_min:x_max] == (i + 1)).astype(np.uint8)
        
        if mask_msml is not None:
            mask_crop = mask_msml[y_min:y_max, x_min:x_max]
        else:
            mask_crop = None
            
        window = rasterio.windows.Window(col_off=x_min, row_off=y_min, width=x_max-x_min, height=y_max-y_min)
        transform_crop = rasterio.windows.transform(window, dem_transform)
        
        tasks.append((dem_crop, rel_crop, transform_crop, point_depth, min_slope, min_height, min_length, mask_crop, slope_chunk_size))
        task_indices.append(i)
        
    if n_processes is None:
        import multiprocessing
        n_processes = max(1, multiprocessing.cpu_count() - 1)
        
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_component, tasks), total=len(tasks), desc="Parallel Progress"))
        
    # Merge results
    full_result = np.zeros_like(rel)
    
    for result_idx, component_idx in enumerate(task_indices):
        sl = slices[component_idx]
        if sl is None: continue
        
        y_slice, x_slice = sl
        y_min = max(0, y_slice.start - buffer_pixels)
        y_max = min(dem_array.shape[0], y_slice.stop + buffer_pixels)
        x_min = max(0, x_slice.start - buffer_pixels)
        x_max = min(dem_array.shape[1], x_slice.stop + buffer_pixels)
        
        full_result[y_min:y_max, x_min:x_max] = np.maximum(
            full_result[y_min:y_max, x_min:x_max], 
            results[result_idx]
        )
        
    akt = utils.polygonize_results(full_result, dem_profile, field="slope").to_crs(epsg=25833)
    return akt

def _group_nearby_components(slices, max_distance_pixels, max_group_size=None):
    """
    Group components whose bounding boxes (expanded by max_distance) overlap.
    Optionally limits group size to prevent over-grouping.
    
    Args:
        slices: List of slice objects from scipy.ndimage.find_objects
        max_distance_pixels: Maximum distance in pixels to expand bounding boxes
        max_group_size: Maximum number of components per group (None = unlimited)
        
    Returns:
        List of groups, where each group is a list of component indices
    """
    if not slices:
        return []
    
    # Expand each slice by max_distance_pixels
    expanded_boxes = []
    for i, sl in enumerate(slices):
        if sl is None:
            expanded_boxes.append(None)
            continue
            
        y_slice, x_slice = sl
        expanded_box = (
            max(0, y_slice.start - max_distance_pixels),
            y_slice.stop + max_distance_pixels,
            max(0, x_slice.start - max_distance_pixels),
            x_slice.stop + max_distance_pixels
        )
        expanded_boxes.append(expanded_box)
    
    # Find overlapping boxes using Union-Find
    parent = list(range(len(slices)))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Check all pairs for overlap
    for i in range(len(expanded_boxes)):
        if expanded_boxes[i] is None:
            continue
        for j in range(i + 1, len(expanded_boxes)):
            if expanded_boxes[j] is None:
                continue
                
            # Check if boxes overlap
            y1_min, y1_max, x1_min, x1_max = expanded_boxes[i]
            y2_min, y2_max, x2_min, x2_max = expanded_boxes[j]
            
            # Boxes overlap if they intersect in both x and y
            x_overlap = not (x1_max <= x2_min or x2_max <= x1_min)
            y_overlap = not (y1_max <= y2_min or y2_max <= y1_min)
            
            if x_overlap and y_overlap:
                union(i, j)
    
    # Group components by their root parent
    groups_dict = {}
    for i in range(len(slices)):
        if slices[i] is None:
            continue
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)
    
    groups = list(groups_dict.values())
    
    # Split oversized groups if max_group_size is specified
    if max_group_size is not None and max_group_size > 0:
        final_groups = []
        for group in groups:
            if len(group) <= max_group_size:
                final_groups.append(group)
            else:
                # Split large group into smaller chunks
                for i in range(0, len(group), max_group_size):
                    final_groups.append(group[i:i + max_group_size])
        return final_groups
    
    return groups


def _process_group(args):
    """
    Helper function for parallel processing of component groups.
    """
    (dem_crop, rel_crop, transform_crop, depth, min_slope, min_height, min_length, mask_crop, slope_chunk_size) = args
    
    # Run retrogression on the group
    result, _ = landslide_retrogression_optimized(
        dem_crop, rel_crop, transform_crop,
        initial_release_depth=depth,
        min_slope=min_slope,
        min_height=min_height,
        min_length=min_length,
        mask=mask_crop,
        verbose=False,
        slope_chunk_size=slope_chunk_size,
    )
    return result


def run_retrogression_parallel_grouped(bounds: tuple,
                                       rel_shape: gpd.GeoDataFrame,
                                       point_depth: float = 0.0,
                                       clip_to_msml=False,
                                       min_slope: float = 1 / 15,
                                       min_height: float = 5,
                                       min_length: float = 75,
                                       max_length: float = 2000,
                                       slope_chunk_size: int = 1000,
                                       custom_raster=None,
                                       n_processes: int = None,
                                       buffer_pixels: int = 50,
                                       grouping_distance: float = None,
                                       max_group_size: int = None) -> gpd.GeoDataFrame:
    """
    Parallel version of run_retrogression with distance-based grouping.
    Groups nearby components that could interact during retrogression, ensuring accurate results.
    
    Args:
        bounds: xmin,xmax,ymin,ymax of the calculation window
        rel_shape: release area as a geodataframe
        point_depth: depth of the source points
        clip_to_msml: whether to clip against MSML
        min_slope: minimum slope of the landslide
        min_height: minimum height for checking the slope criterion
        min_length: minimum length of the landslide
        max_length: maximum length of the landslide (fallback for grouping distance)
        slope_chunk_size: chunk size for slope calculation (default 1000) to keep memory/time down while preserving results
        custom_raster: custom raster to use for the calculation
        n_processes: number of parallel processes (default: auto)
        buffer_pixels: buffer around each group in pixels
        grouping_distance: custom grouping distance in meters (default: max_length)
                          Use smaller values (e.g., 300-500m) for better parallelization
        max_group_size: maximum components per group (default: None = unlimited)
                        Use to prevent over-grouping (e.g., 10-50 components)
        
    Returns:
        akt (gpd.GeoDataFrame): propagated release area
    """
    if custom_raster is None:
        dem_data = utils.get_hoydedata(bounds)
    else:
        dem_data = utils.generate_windows(custom_raster)

    dem_array = dem_data["full_array"]
    dem_profile = dem_data["profile"]
    dem_transform = dem_profile["transform"]
    
    # Get resolution for distance calculation
    resolution = abs(dem_transform[0])
    
    # Calculate buffer pixels if not provided
    if buffer_pixels is None:
        # Buffer should be at least max_length to allow full propagation
        # We add a small safety margin (e.g. 10%)
        buffer_pixels = int((max_length * 1.1) / resolution)
        # print(f"Calculated buffer: {buffer_pixels} pixels ({buffer_pixels * resolution:.1f} m)")
    
    # Use custom grouping distance if provided, otherwise fall back to max_length
    actual_grouping_distance = grouping_distance if grouping_distance is not None else max_length
    max_distance_pixels = int(actual_grouping_distance / resolution)

    if clip_to_msml:
        mask_gpd = utils.get_msml_mask((bounds[0], bounds[2], bounds[1], bounds[3]))
        mask_msml = utils.rasterize_shape(mask_gpd, dem_profile)
    else:
        mask_msml = None

    # Rasterize release shape
    rel = utils.rasterize_shape(rel_shape, dem_profile)
    
    # Label connected components
    labeled_array, num_features = label(rel)
    
    if num_features == 0:
        return gpd.GeoDataFrame(columns=["geometry", "slope"], crs=25833)
    
    slices = find_objects(labeled_array)
    
    # Group nearby components with optional size limit
    groups = _group_nearby_components(slices, max_distance_pixels, max_group_size=max_group_size)
    
    # Report grouping statistics
    avg_group_size = sum(len(g) for g in groups) / len(groups) if groups else 0
    max_actual_group_size = max(len(g) for g in groups) if groups else 0
    print(f"Found {num_features} components, grouped into {len(groups)} groups for parallel processing")
    print(f"  - Average group size: {avg_group_size:.1f} components")
    print(f"  - Largest group: {max_actual_group_size} components")
    print(f"  - Grouping distance: {actual_grouping_distance:.0f}m ({max_distance_pixels} pixels)")
    
    # Prepare tasks for each group
    tasks = []
    task_groups = []  # Track which group each task corresponds to
    
    for group_idx, group in enumerate(groups):
        # Find combined bounding box for the entire group
        y_min = min(dem_array.shape[0], *[slices[i][0].start for i in group if slices[i] is not None])
        y_max = max(0, *[slices[i][0].stop for i in group if slices[i] is not None])
        x_min = min(dem_array.shape[1], *[slices[i][1].start for i in group if slices[i] is not None])
        x_max = max(0, *[slices[i][1].stop for i in group if slices[i] is not None])
        
        # Add buffer
        y_min = max(0, y_min - buffer_pixels)
        y_max = min(dem_array.shape[0], y_max + buffer_pixels)
        x_min = max(0, x_min - buffer_pixels)
        x_max = min(dem_array.shape[1], x_max + buffer_pixels)
        
        # Crop data for the group
        dem_crop = dem_array[y_min:y_max, x_min:x_max]
        
        # Create release mask for all components in the group
        rel_crop = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        for comp_idx in group:
            if slices[comp_idx] is None:
                continue
            # Extract component and place it in the cropped array
            comp_mask = (labeled_array == (comp_idx + 1)).astype(np.uint8)
            rel_crop = np.maximum(rel_crop, comp_mask[y_min:y_max, x_min:x_max])
        
        if mask_msml is not None:
            mask_crop = mask_msml[y_min:y_max, x_min:x_max]
        else:
            mask_crop = None
            
        # Calculate transform for the crop
        window = rasterio.windows.Window(col_off=x_min, row_off=y_min, width=x_max-x_min, height=y_max-y_min)
        transform_crop = rasterio.windows.transform(window, dem_transform)

        tasks.append((dem_crop, rel_crop, transform_crop, point_depth, min_slope, min_height, min_length, mask_crop, slope_chunk_size))
        task_groups.append((group_idx, y_min, y_max, x_min, x_max))
        
    # Run in parallel
    if n_processes is None:
        import multiprocessing
        n_processes = max(1, multiprocessing.cpu_count() - 1)
        
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(_process_group, tasks), total=len(tasks), desc="Parallel Progress (Grouped)"))
        
    # Merge results
    full_result = np.zeros_like(rel)
    
    for result_idx, (group_idx, y_min, y_max, x_min, x_max) in enumerate(task_groups):
        # Add result back to full array
        full_result[y_min:y_max, x_min:x_max] = np.maximum(
            full_result[y_min:y_max, x_min:x_max], 
            results[result_idx]
        )
        
    # Polygonize
    akt = utils.polygonize_results(full_result, dem_profile, field="slope").to_crs(epsg=25833)
    
    return akt


def run_retrogression_parallel_adaptive(bounds: tuple,
                                         rel_shape: gpd.GeoDataFrame,
                                         point_depth: float = 0.0,
                                         clip_to_msml=False,
                                         min_slope: float = 1 / 15,
                                         min_height: float = 5,
                                         min_length: float = 75,
                                         slope_chunk_size: int = 1000,
                                         custom_raster=None,
                                         n_processes: int = None,
                                         buffer_pixels: int = 50,
                                         speed_priority: str = 'balanced') -> gpd.GeoDataFrame:
    """
    Adaptive parallel retrogression that automatically chooses the best strategy.
    
    WARNING: Parallel methods work well ONLY for datasets with well-separated components.
    For dense stream networks where retrogression zones interact, use run_retrogression()
    (serial method) instead for accurate results.
    
    This function analyzes the release area components and intelligently decides
    how to group them for optimal parallelization while maintaining accuracy.
    
    Args:
        bounds: xmin,xmax,ymin,ymax of the calculation window
        rel_shape: release area as a geodataframe
        point_depth: depth of the source points
        clip_to_msml: whether to clip against MSML
        min_slope: minimum slope of the landslide
        min_height: minimum height for checking the slope criterion
        min_length: minimum length of the landslide
        slope_chunk_size: chunk size for slope calculation
        custom_raster: custom raster to use for the calculation
        n_processes: number of parallel processes (default: auto)
        buffer_pixels: buffer around each group in pixels
        speed_priority: Strategy for balancing speed vs accuracy
            - 'speed': Aggressive, may miss interactions (300m, for well-separated components)
            - 'balanced': Moderate (1500m grouping, safer but may still have errors)
            - 'accuracy': Conservative (3000m grouping, closest to serial but may still differ)
            - 'serial': Just use serial method (100% accurate, recommended for interacting components)
        
    Returns:
        akt (gpd.GeoDataFrame): propagated release area
    """
    # Strategy parameters - MUCH more conservative now
    strategies = {
        'speed': {
            'grouping_distance': 300,   # Only for truly isolated components
            'max_group_size': None,     # No splitting
            'description': 'Fast (300m grouping) - USE ONLY for well-separated components'
        },
        'balanced': {
            'grouping_distance': 1500,  # More conservative
            'max_group_size': None,     # No splitting to avoid breaking interactions
            'description': 'Balanced (1500m grouping) - May still have errors'
        },
        'accuracy': {
            'grouping_distance': 3000,  # Very conservative
            'max_group_size': None,     # No splitting
            'description': 'Accurate (3000m grouping) - Closest to serial'
        },
        'serial': {
            'use_serial': True,
            'description': 'Serial (no parallelization) - 100% accurate'
        }
    }
    
    if speed_priority not in strategies:
        print(f"[WARNING] Unknown speed_priority '{speed_priority}', using 'balanced'")
        speed_priority = 'balanced'
    
    strategy = strategies[speed_priority]
    
    # If serial mode requested, just use the serial implementation
    if strategy.get('use_serial'):
        print(f"\n{'='*70}")
        print(f"Using Serial Method (100% Accurate)")
        print(f"{'='*70}\n")
        return run_retrogression(
            bounds=bounds,
            rel_shape=rel_shape,
            point_depth=point_depth,
            clip_to_msml=clip_to_msml,
            min_slope=min_slope,
            min_height=min_height,
            min_length=min_length,
            slope_chunk_size=slope_chunk_size,
            custom_raster=custom_raster,
            return_animation=False,
            verbose=False
        )
    
    print(f"\n{'='*70}")
    print(f"Adaptive Parallel Retrogression - {strategy['description']}")
    print(f"{'='*70}\n")
    
    # Calculate safe buffer size based on max_length
    # We need to ensure the crop is large enough for the landslide to propagate fully.
    # max_length is in meters. We need pixels.
    # We don't have resolution here easily without opening DEM, but we can estimate or pass it.
    # Actually, run_retrogression_parallel_grouped calculates resolution.
    # Let's pass a flag or large enough buffer.
    # Better yet, let run_retrogression_parallel_grouped handle the buffer calculation if not provided.
    
    # Use the grouped parallel implementation with adaptive parameters
    return run_retrogression_parallel_grouped(
        bounds=bounds,
        rel_shape=rel_shape,
        point_depth=point_depth,
        clip_to_msml=clip_to_msml,
        min_slope=min_slope,
        min_height=min_height,
        min_length=min_length,
        slope_chunk_size=slope_chunk_size,
        custom_raster=custom_raster,
        n_processes=n_processes,
        buffer_pixels=None, # Let the function calculate it based on max_length
        grouping_distance=strategy['grouping_distance'],
        max_group_size=strategy['max_group_size']
    )
