"""
Shared utilities for geometry construction.
All angles are in radians internally unless noted.
"""
from __future__ import annotations
import numpy as np

# Expose utilities (including internal helpers) for package-internal star-imports.
# Note: keeping leading-underscore names here allows internal modules to use
# `from .utils import *` while the package top-level (`geometry/__init__.py`) avoids
# importing utils.* so external `from geometry import *` won't expose these names.
__all__ = [
    'n_per_ring',
    '_spacing_ring',
    '_ring_xy',
    '_parse_interval_deg',
    '_discretize_arc_by_dl',
    '_resolve_axis_or_plane',
    '_transform_coordinate',
    '_arange0_quantized',
    'get_wall_ID',
]


def n_per_ring(r, d, phi_ring=2 * np.pi) -> np.ndarray | int:
    """
    Compute the number of points for a ring at a given radius with specified spacing.

    Args:
        r (float or np.ndarray): Radius of the ring(s).
        d (float): Desired spacing between points.
        phi_ring (float): Total angle of the ring in radians (default is 2π for a full circle).

    Returns:
        np.ndarray or int: Number of points per ring. Returns an integer if `r` is scalar.
    """
    r = np.asarray(r, dtype=float).flatten()
    n = np.ones_like(r, dtype=float)
    mask = r != 0
    # Calculate the number of points based on chord length and central angle.
    n[mask] = phi_ring / np.arccos(1 - 0.5 * (d / r[mask]) ** 2)
    if n.size == 1:
        return int(round(n.item()))
    return np.round(n).astype(int)


def _spacing_ring(r, n, phi_ring=2 * np.pi) -> np.ndarray:
    """
    Compute the spacing between points on a ring given the number of points.

    Args:
        r (float or np.ndarray): Radius of the ring(s).
        n (int or np.ndarray): Number of points on the ring(s).
        phi_ring (float): Total angle of the ring in radians (default is 2π for a full circle).

    Returns:
        np.ndarray: Spacing between neighboring points on the ring.
    """
    r = np.asarray(r, dtype=float)
    return r * np.sqrt(2 * (1 - np.cos(phi_ring / n)))


def _ring_xy(n: int, r: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Cartesian coordinates of points on a ring in the xOy plane.

    Args:
        n (int): Number of points on the ring.
        r (float): Radius of the ring.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates of the points.
    """
    angles = np.linspace(0, 2 * np.pi, int(n), endpoint=False)
    return r * np.cos(angles), r * np.sin(angles)


def _parse_interval_deg(interval: str) -> tuple[float, float, bool, bool, float]:
    """
    Parse an angular interval in degrees into its components.

    Args:
        interval (str): Interval string in the format '[a,b)'.

    Returns:
        tuple: (min_deg, max_deg, incl_min, incl_max, total_deg)
            - min_deg (float): Start of the interval in degrees.
            - max_deg (float): End of the interval in degrees.
            - incl_min (bool): Whether the start is inclusive.
            - incl_max (bool): Whether the end is inclusive.
            - total_deg (float): Total angular span of the interval.

    Raises:
        ValueError: If the interval format is invalid or the angular span exceeds 360 degrees.
    """
    if len(interval) < 5 or interval[0] not in '([{' or interval[-1] not in ')]}':
        raise ValueError("Interval must look like '[a,b)'")
    incl_min = interval[0] == '['
    incl_max = interval[-1] == ']'
    a_str, b_str = interval[1:-1].split(',')
    a, b = float(a_str), float(b_str)
    if b < a:
        raise ValueError('Upper bound must be >= lower bound')
    total = b - a
    if total > 360 + 1e-9:
        raise ValueError('Angular span cannot exceed 360 degrees')
    return a, b, incl_min, incl_max, total


def _discretize_arc_by_dl(r: float, dl: float, a_deg: float,
                         b_deg: float, incl_min: bool, incl_max: bool) -> np.ndarray:
    """
    Generate angular samples for an arc based on spacing.

    Args:
        r (float): Radius of the arc.
        dl (float): Desired spacing between points.
        a_deg (float): Start angle of the arc in degrees.
        b_deg (float): End angle of the arc in degrees.
        incl_min (bool): Whether to include the start angle.
        incl_max (bool): Whether to include the end angle.

    Returns:
        np.ndarray: Array of angular samples in radians.
    """
    a = np.deg2rad(a_deg)
    b = np.deg2rad(b_deg)
    phi_tot = b - a
    n_phi = int(n_per_ring(r, dl, phi_tot))
    if n_phi <= 0:
        return np.array([], dtype=float)
    d_phi = phi_tot / n_phi
    start = int(not incl_min)
    stop = n_phi + int(incl_max)
    return d_phi * np.arange(start, stop) + a


def _resolve_axis_or_plane(axis: str | None = None, plane: str | None = None) -> str:
    """
    Resolve the 'up' axis based on the specified axis or plane.

    Args:
        axis (str | None): Axis name ('x', 'y', or 'z').
        plane (str | None): Plane name ('XOY', 'YOZ', or 'XOZ').

    Returns:
        str: Resolved axis name ('x', 'y', or 'z').

    Raises:
        KeyError: If neither or both axis and plane are specified.
        ValueError: If the axis or plane name is invalid.
    """
    plane2axis = {'XOY': 'z', 'YOZ': 'x', 'XOZ': 'y'}
    if (axis is None) == (plane is None):
        raise KeyError('Specify exactly one of axis or plane')
    if axis is not None:
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be one of 'x','y','z'")
        return axis
    if plane not in plane2axis:
        raise ValueError("plane must be one of 'XOY','YOZ','XOZ'")
    return plane2axis[plane]


def _transform_coordinate(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                         *, axis: str | None = None, plane: str | None = None):
    """
    Transform coordinates to align with the specified axis or plane.

    Args:
        xs (np.ndarray): X-coordinates.
        ys (np.ndarray): Y-coordinates.
        zs (np.ndarray): Z-coordinates.
        axis (str | None): Target axis ('x', 'y', or 'z').
        plane (str | None): Target plane ('XOY', 'YOZ', or 'XOZ').

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Transformed coordinates.
    """
    axis_up = _resolve_axis_or_plane(axis, plane)
    if axis_up == 'z':
        xs, ys, zs = zs, xs, ys
    elif axis_up == 'x':
        xs, ys, zs = ys, zs, xs
    return xs, ys, zs


def _arange0_quantized(end: float, dl: float) -> np.ndarray:
    """
    Generate a quantized range from 0 to `end` with a specified step size.

    Args:
        end (float): End value of the range.
        dl (float): Step size.

    Returns:
        np.ndarray: Quantized range array.
    """
    return np.arange(0, end + dl * 0.1, dl).round(6)


def get_wall_ID(i, j, n_per_ring, smallest_ID=1):
    """
    Compute the ID of a cylinder wall based on its ring and axis indices.

    Args:
        i (int): Index on the ring.
        j (int): Index on the axis.
        n_per_ring (int): Number of points per ring.
        smallest_ID (int): Starting ID value (default is 1).

    Returns:
        int: Computed wall ID.
    """
    if i > n_per_ring:
        i = i % n_per_ring
    return (j - 1) * n_per_ring + i + smallest_ID - 1