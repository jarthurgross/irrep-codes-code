from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def complex_hinton(
    matrix: np.ndarray,
    cmap: mpl.colors.Colormap = plt.cm.hsv,
    normalization: Optional[float] = None,
    max_square_frac: float = 0.9375,
    ax: Optional[plt.Axes] = None,
) -> None:
    ax = ax if ax is not None else plt.gca()
    normalized_angles = (np.angle(matrix) % (2 * np.pi)) / (2 * np.pi)
    lengths = np.sqrt(np.abs(matrix))
    if normalization is None:
        normalization = np.max(lengths)
    normalized_lengths = max_square_frac * lengths / normalization
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
    border_width: float = 1.0,
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
        linewidth=border_width,
    )
    ax.plot(
        outer_radius * np.cos(angles),
        outer_radius * np.sin(angles),
        color=border_color,
        linewidth=border_width,
    )
    ax.set_aspect("equal")
    ax.axis("off")


def complex_hinton_plot(
    matrix: np.ndarray,
    cmap: plt.Axes = plt.cm.hsv,
    normalization: Optional[float] = None,
    max_square_frac: float = 0.9375,
    border_color: str = "k",
    axs: Optional[Tuple[plt.Axes, plt.Axes]] = None,
) -> None:
    if axs is None:
        _, axs = plt.subplots(ncols=2)
    complex_hinton(matrix=matrix, cmap=cmap, normalization=normalization, ax=axs[0])
    cyclic_cbar(cmap=cmap, border_color=border_color, ax=axs[1])


def make_hexagon(
    center: Tuple[float, float],
    corner: Tuple[float, float],
    **kwargs,
) -> mpl.patches.Polygon:
    """Make a hexagon at a particular position and orientation.

    Parameters
    ----------
    center
        The x/y coordinates for the center of the hexagon.
    corner
        The x/y coordinates for one of the corners of the hexagon.

    Returns
    -------
        Hexagon represented as a polygon.

    """
    displacement = np.array(corner) - np.array(center)
    corner_angle = np.arctan2(displacement[1], displacement[0])
    angles = np.pi * np.arange(6) / 3 + corner_angle
    radius = np.linalg.norm(displacement)
    unit_xy = np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=-1,
    )
    return plt.Polygon(
        np.array(center) + radius * unit_xy,
        **kwargs,
    )


def triplet_coord_to_xy(triplet_coord: np.ndarray, angle0: float = 0., counterclockwise: bool = True):
    """Project an x/y/z coordinate orthogonal to the x=y=z line.

    Coordinates are projected with a scaling factor such that the point
    (1, 0, 0) will project to a point one unit from the origin of the plane.

    Parameters
    ----------
    triplet_coord
        An array of x/y/z coordinates with the last axis indexing x/y/z.
    angle0
        The angle of displacement in the plane for the vector (1, 0, 0).
    counterclockwise
        Whether the angle of displacement for the vector (0, 1, 0) is rotated
        120 degrees clockwise from the (1, 0, 0) angle (if False it is rotated
        counterclockwise)

    Returns
    -------
        An array of x/y coordinates for the projected points, whose last axis
        indexes x/y.

    """
    sign = 1 if counterclockwise else -1
    angles = angle0 + sign * 2 * np.pi * np.arange(3) / 3
    vecs = np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=-1,
    )
    return triplet_coord @ vecs


def complex_hex_hinton(
    values: np.ndarray,
    triplet_coords: np.ndarray,
    cmap: mpl.colors.Colormap = plt.cm.hsv,
    normalization: Optional[float] = None,
    max_hex_frac: float = 0.9375,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Plot a complex Hinton diagram with hexagons centered at given points.

    Centers of the hexagons are provided in triplet form such that triangular
    grids may be naturally supplied.

    Parameters
    ----------
    values
        Complex values to be represented.
    triplet_coords
        x/y/z coordinates specifying where to center the hexagons. These values
        will be projected to a plane according to the default settings for
        `triplet_coord_to_xy`.
    """
    ax = ax if ax is not None else plt.gca()
    normalized_angles = (np.angle(values) % (2 * np.pi)) / (2 * np.pi)
    lengths = np.sqrt(np.abs(values))
    if normalization is None:
        normalization = np.max(lengths)
    normalized_lengths = max_hex_frac * lengths / normalization
    xys = triplet_coord_to_xy(triplet_coords, angle0=np.pi / 2)
    for xy, length, angle in zip(xys, normalized_lengths, normalized_angles):
        color = cmap(angle)
        hexagon = make_hexagon(
            center=tuple(xy),
            corner=xy + length * np.array([0, 1]),
            facecolor=color,
            edgecolor=(0, 0, 0, 0),
        )
        ax.add_patch(hexagon)
    ax.patch.set_facecolor((0, 0, 0, 0))
    ax.autoscale_view()
    ax.set_aspect("equal")
