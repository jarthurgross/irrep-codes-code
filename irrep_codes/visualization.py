from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def complex_hinton(
    matrix: np.ndarray,
    cmap: mpl.colors.Colormap = plt.cm.hsv,
    ax: Optional[plt.Axes] = None,
) -> None:
    ax = ax if ax is not None else plt.gca()
    normalized_angles = (np.angle(matrix) % (2 * np.pi)) / (2 * np.pi)
    lengths = np.sqrt(np.abs(matrix))
    normalized_lengths = lengths / np.max(lengths)
    num_rows, _ = matrix.shape
    for (row, col), length in np.ndenumerate(normalized_lengths):
        color = cmap(normalized_angles[row, col])
        rect = plt.Rectangle(
            [col - length / 2, row - length / 2],
            length,
            length,
            facecolor=color,
            edgecolor=(0, 0, 0, 0),
        )
        ax.add_patch(rect)
    ax.patch.set_facecolor((0, 0, 0, 0))
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_aspect("equal")


def make_annulus_triangulation(
    num_angle_points: int,
    inner_radius: float,
    outer_radius: float,
) -> mpl.tri.Triangulation:
    angles = np.linspace(0, 2 * np.pi, num_angle_points)
    unit_xs = np.cos(angles)
    unit_ys = np.sin(angles)
    inner_xs = inner_radius * unit_xs
    inner_ys = inner_radius * unit_ys
    outer_xs = outer_radius * unit_xs
    outer_ys = outer_radius * unit_ys
    xs = np.hstack([inner_xs, outer_xs])
    ys = np.hstack([inner_ys, outer_ys])
    inner_triangles = np.array(
        [[j + 1, j, num_angle_points + j] for j in range(num_angle_points - 1)]
    )
    outer_triangles = np.array(
        [
            [num_angle_points + j, num_angle_points + j + 1, j + 1]
            for j in range(num_angle_points - 1)
        ]
    )
    triangles = np.vstack([inner_triangles, outer_triangles])
    return mpl.tri.Triangulation(
        x=xs,
        y=ys,
        triangles=triangles,
    )


def cyclic_cbar(
    num_angle_points: int = 2**8 + 1,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    cmap: mpl.colors.Colormap = plt.cm.hsv,
    border_color: str = "k",
    ax: Optional[plt.Axes] = None,
) -> None:
    triangulation = make_annulus_triangulation(
        num_angle_points=num_angle_points,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )
    angles = np.arctan2(triangulation.y, triangulation.x)
    normalized_angles = (angles % (2 * np.pi)) / (2 * np.pi)
    ax = ax if ax is not None else plt.gca()
    ax.tripcolor(
        triangulation,
        normalized_angles,
        cmap=cmap,
        rasterized=True,
        shading="gouraud",
    )
    angles = np.linspace(0, 2 * np.pi, num_angle_points)
    ax.plot(
        inner_radius * np.cos(angles),
        inner_radius * np.sin(angles),
        color=border_color,
    )
    ax.plot(
        outer_radius * np.cos(angles),
        outer_radius * np.sin(angles),
        color=border_color,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def complex_hinton_plot(
    matrix: np.ndarray,
    cmap: plt.Axes = plt.cm.hsv,
    border_color: str = "k",
    axs: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> None:
    if axs is None:
        _, axs = plt.subplots(ncols=2)
    complex_hinton(matrix=matrix, cmap=cmap, ax=axs[0])
    cyclic_cbar(cmap=cmap, border_color=border_color, ax=axs[1])
