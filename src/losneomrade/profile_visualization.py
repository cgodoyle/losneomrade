from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from .utils import plot_geometries

logger = logging.getLogger(__name__)


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

    import matplotlib.pyplot as plt

    from .profile_retrogression import compute_slope_line_terrain_intersection

    height_diff, exit_dist, intersects, start_offset = compute_slope_line_terrain_intersection(
        distances,
        elevations,
        initial_depth,
        slope_ratio,
        max_distance,
        start_tolerance,
    )

    start_elevation = elevations[0] - initial_depth
    line_elevations = start_elevation + slope_ratio * (distances - start_offset)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(distances, elevations, "k-", linewidth=2, label="Terrain", zorder=3)
    ax.fill_between(
        distances,
        elevations,
        elevations.min() - 10,
        alpha=0.3,
        color="brown",
        label="Ground",
    )

    mask_valid = distances >= start_offset
    mask_within_max = (distances - start_offset) <= max_distance
    mask_plot = mask_valid & mask_within_max
    ax.plot(
        distances[mask_plot],
        line_elevations[mask_plot],
        "r--",
        linewidth=2,
        label=f"Slope Line (1:{int(1 / slope_ratio)} V:H)",
        zorder=4,
    )

    if start_tolerance > 0:
        ax.axvspan(
            0,
            start_tolerance,
            alpha=0.1,
            color="green",
            label=f"Start Tolerance ({start_tolerance}m)",
        )

    ax.plot(
        start_offset,
        start_elevation,
        "go",
        markersize=10,
        label=f"Optimal Start (offset={start_offset:.1f}m, depth={initial_depth}m)",
        zorder=5,
    )

    below_terrain = line_elevations < elevations
    intersection_zone = below_terrain & mask_plot
    if np.any(intersection_zone):
        ax.fill_between(
            distances[intersection_zone],
            line_elevations[intersection_zone],
            elevations[intersection_zone],
            alpha=0.5,
            color="red",
            label="Intersection Zone",
        )

    if intersects:
        exit_idx = np.argmin(np.abs(distances - exit_dist))
        ax.plot(
            exit_dist,
            elevations[exit_idx],
            "r*",
            markersize=15,
            label=f"Exit Point ({exit_dist:.1f}m)",
            zorder=5,
        )
        ax.annotate(
            f"Exit: {exit_dist:.1f}m\nΔh: {height_diff:.2f}m",
            xy=(exit_dist, elevations[exit_idx]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    ax.axvline(
        start_offset + max_distance,
        color="gray",
        linestyle=":",
        label=f"Max Distance ({max_distance}m)",
    )
    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("Elevation (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    status_text = f"MAX Δh: {height_diff:.2f}m" if intersects else "NO INTERSECTION"
    status_color = "red" if intersects else "green"
    ax.text(
        0.02,
        0.98,
        status_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=status_color,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


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

    import matplotlib.pyplot as plt

    num_profiles = len(profiles)
    cols = min(4, num_profiles)
    rows = (num_profiles + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes_grid = np.atleast_2d(axes)
    logger.debug(len(profiles))

    for index, (distances, elevations, retro_distance, slope_line_plot, height) in enumerate(
        zip(
            distances_list,
            elevations_list,
            retro_distance_list,
            slope_line_plot_list,
            max_heights_list,
        ),
    ):
        row = index // cols
        col = index % cols
        ax = axes_grid[row, col]

        ax.plot(
            distances,
            elevations,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Elevation",
        )
        ax.plot(
            distances,
            slope_line_plot,
            linestyle="--",
            color="red",
            label=f"1:{int(1 / slope_ratio)} slope",
        )

        if retro_distance is not None:
            retro_elevation = np.interp(retro_distance, distances, elevations)
            ax.axvline(
                x=retro_distance,
                color="green",
                linestyle=":",
                linewidth=2,
                label=f"Retrogression: {retro_distance:.1f}m",
            )
            ax.plot(
                retro_distance,
                retro_elevation,
                "go",
                markersize=8,
                markeredgecolor="darkgreen",
                markeredgewidth=2,
            )

            x_limit = max(np.ceil((retro_distance * 1.2) / 100) * 100, 50)
            ax.set_xlim(0, x_limit)

            mask_xlim = np.array(distances) <= x_limit
            elevations_in_range = np.array(elevations)[mask_xlim]
            elevations_valid = elevations_in_range[~np.isnan(elevations_in_range)]

            if len(elevations_valid) > 0:
                y_range = elevations_valid.max() - elevations_valid.min()
                y_min = elevations_valid.min() - y_range * 0.2
                y_max = elevations_valid.max() + y_range * 0.2
                ax.set_ylim(y_min, y_max)

            pos_x = retro_distance
            pos_y = retro_elevation + (max(elevations) - min(elevations)) * 0.05
            pos_y = min(pos_y, y_max)
            if np.isnan(pos_y) or np.isinf(pos_y) or not np.isreal(pos_y):
                pos_y = float(np.mean(ax.get_ylim()))

            ax.text(
                pos_x,
                pos_y,
                f"d={retro_distance:.1f}m\nH={height:.1f}m",
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )

            if retro_distance == 0:
                ax.text(
                    x_limit / 2,
                    (y_min + y_max) / 2,
                    "No retrogression",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    color="red",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
                )

        ax.set_title(f"Profile {index + 1}")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for index in range(num_profiles, rows * cols):
        row = index // cols
        col = index % cols
        ax = axes_grid[row, col]
        ax.set_visible(False)

    plt.tight_layout()

    ax = gpd.GeoDataFrame(geometry=points_coords).plot(
        color="red",
        markersize=50,
        alpha=0.7,
        label="Retrogression Points",
    )
    return_gdf.plot(ax=ax, color="blue", alpha=0.3, edgecolor="black", label="Release Envelope")
    profile_gdf = gpd.clip(gpd.GeoDataFrame(geometry=profiles), return_gdf.buffer(20))
    profile_gdf.plot(ax=ax, alpha=0.7, label="Profiles")

    for index, dem_profile in zip(profile_gdf.index, profile_gdf.geometry.to_list()):
        ax.annotate(
            f"{index + 1}",
            list(dem_profile.coords)[1],
            fontsize=8,
            ha="center",
            va="center",
        )

    plot_geometries(line, ax, color="red")
    ax.legend()
    plt.show()
