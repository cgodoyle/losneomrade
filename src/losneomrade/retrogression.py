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
from scipy.ndimage import binary_dilation
from tqdm.notebook import tqdm

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

    release, anim = landslide_retrogression(
        dem_array, rel, dem_profile["transform"], initial_release_depth=point_depth,
        min_slope=min_slope, min_height=min_height, min_length=min_length, mask=mask_msml,
        verbose=verbose)

    akt = utils.polygonize_results(release, dem_profile, field="slope").to_crs(epsg=25833)
    if return_animation:
        akt = (akt, anim)
    return akt


def run_retrogression_with_initial_landslide(
        bounds: tuple,
        rel_shape: gpd.GeoDataFrame,
        point_depth: float = 0.0,
        clip_to_msml=False,
        ini_slope: float = 1 / 4,
        retro_slope: float = 1 / 15,
        min_height: float = 5,
        min_length: float = 75,
        custom_raster=None,
        return_animation=False,
        
):
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

    min_length_first = min_length * ini_slope
    min_length_second = min_length - min_length_first

    release_first, animation_first = landslide_retrogression(
        dem_array, 
        rel, dem_profile["transform"], 
        initial_release_depth=point_depth,
        min_slope=ini_slope, 
        min_height=min_height, 
        min_length=min_length_first, 
        mask=mask_msml,
        verbose=False)
    
    if np.all(release_first == rel):
        akt = gpd.GeoDataFrame(columns=["geometry", "slope"], crs=25833)
        animation_second = []

    else:
        release_second, animation_second = landslide_retrogression(
            dem=dem_array,
            initial_release=release_first,
            dem_transform=dem_profile["transform"],
            min_slope=retro_slope,
            min_height=0,
            min_length=min_length_second,
            max_length=2000,
            initial_release_depth=0,
            mask=mask_msml,
            verbose=False
        )
        

        first_release = utils.polygonize_results(release_first, dem_profile, field="slope").to_crs(epsg=25833)
        first_release["slope"] = ini_slope
        second_release = utils.polygonize_results(release_second, dem_profile, field="slope").to_crs(epsg=25833)
        second_release["slope"] = retro_slope
        akt = pd.concat([first_release, second_release], ignore_index=True)


    if return_animation:
        animation = animation_first + [animation_first[-1] for _ in range(100)] + animation_second
        akt = (akt, animation)
    return akt

  

def landslide_retrogression(dem: np.ndarray,
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

    with tqdm(total=0, desc="iterations", disable=not verbose) as pbar:
        while n_iter < max_iter:

            buffered = create_buffer(release, 1)
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

            if mask is not None:
                release_after[mask == 0] = 0

            if np.all(release.astype(bool) == release_after.astype(bool)) and n_iter > min_iter:
                break

            release = release_after.copy()

            animation.append(release_after)

            n_iter += 1
            pbar.update(1)

    if np.all(release == initial_release+create_buffer(initial_release, min_iter)):
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
    buffer = ((dilated_image - image) > 0).astype(bool)

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
