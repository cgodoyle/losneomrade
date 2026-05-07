from __future__ import annotations

import io
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def animate_landslide_retrogresion(
    animation: list[np.ndarray],
    dem: np.ndarray,
    frame_step: int | None = None,
) -> Any:
    """Create a plotly animation of the landslide retrogression."""

    import plotly.graph_objects as go

    logger.info("Animating landslide retrogression")

    if frame_step is None:
        frame_step = len(animation) // 5 if len(animation) // 5 > 1 else 2

    color_red = "rgba(255, 0, 0, 0.5)"
    color_white = "rgba(255, 255, 255, 0.0)"
    basemap = hillshade_img(dem, 1)
    fig_data = [basemap, go.Heatmap(z=animation[0], colorscale=[[0, color_white], [1, color_red]], showscale=False)]
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            title="Step 0",
            updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])],
        ),
    )

    frames = [
        go.Frame(
            data=[basemap, go.Heatmap(z=animation[i], colorscale=[[0, color_white], [1, color_red]], showscale=False)],
            layout=go.Layout(title_text=f"Step {i}"),
        )
        for i in range(1, len(animation), frame_step)
    ]
    frames.append(
        go.Frame(
            data=[basemap, go.Heatmap(z=animation[-1], colorscale=[[0, color_white], [1, color_red]], showscale=False)],
            layout=go.Layout(title_text=f"Step {len(animation)}"),
        )
    )
    fig.frames = frames

    height, width = dem.shape

    fig.update_xaxes(scaleanchor="y")
    fig.update_yaxes(scaleratio=1, autorange="reversed")
    fig.update_layout(xaxis_range=[0, width], yaxis_range=[0, height])
    fig.update_layout(
        width=500,
        height=500,
        coloraxis_showscale=False,
        plot_bgcolor=color_white,
    )

    return fig


def hillshade_img(dem_array: np.ndarray, ve: float = 1) -> Any:
    """Create a plotly Image object of a hillshade."""

    import plotly.graph_objects as go
    from matplotlib import pyplot as plt
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=315, altdeg=45)
    hilsh = ls.shade(dem_array, vert_exag=ve, blend_mode="hsv", cmap=plt.get_cmap("gray"), dx=5, dy=5)
    img = np.array((255 * hilsh[:, :, :3] + 0.5), int)
    return go.Image(z=img)


def plot_hillshade_overlay(
    dem: np.ndarray,
    overlay: np.ndarray,
    ve: float = 1,
    alpha: float = 0.4,
    res: float = 5,
    figsize: tuple[float, float] = (10, 10),
) -> Any:
    """Plot a hillshade overlaid with a binary overlay."""

    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from matplotlib.colors import LightSource

    current_backend = plt.get_backend()
    plt.switch_backend("Agg")

    cmap = mcolors.ListedColormap(["none", "red"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ls = LightSource(azdeg=315, altdeg=45)

    fig, ax = plt.subplots(figsize=figsize)
    _ = ax.imshow(ls.hillshade(dem, vert_exag=ve, dx=res, dy=res), cmap="gray")
    _ = ax.imshow(overlay, cmap=cmap, norm=norm, alpha=alpha)

    plt.switch_backend(current_backend)

    return fig


def gen_animation(
    dem: np.ndarray,
    animation: list[np.ndarray],
    skip_frames: int = 10,
    filename: str | None = None,
) -> list:
    """Generate a GIF animation from retrogression frames."""

    from matplotlib import pyplot as plt
    from PIL import Image

    fig_list = [plot_hillshade_overlay(dem, ani) for ani in animation[::skip_frames]]
    frames = []

    for fig in fig_list:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)

        img = Image.open(buffer)
        frames.append(img.copy())

        plt.clf()

    if filename is not None:
        frames[0].save(filename, save_all=True, append_images=frames[1:], loop=0, duration=200)
    return frames


def save_frames(dem_array: np.ndarray, animation: list[np.ndarray], out_dir: str, skip_frames: int = 10) -> None:
    """Save retrogression animation frames as PNG images."""

    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm

    current_backend = plt.get_backend()
    plt.switch_backend("Agg")
    os.makedirs(out_dir, exist_ok=True)
    n_frames = len(animation[::skip_frames])
    for ii, ani in tqdm(enumerate(animation[::skip_frames]), total=n_frames, desc="saving frames"):
        fig = plot_hillshade_overlay(dem_array, ani)
        fig.savefig(f"{out_dir}\\gif_frame_{ii}.png")
        fig.clf()
    plt.switch_backend(current_backend)
