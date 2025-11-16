from __future__ import annotations
import numpy as np
from .base import Geometry
from scipy.spatial import KDTree
from typing import Iterable


class Union(Geometry):
    """Concatenate multiple geometries."""

    def __init__(self, geometries, name=None):
        """
        Initialize a Union object that concatenates multiple geometries.
        Users had better ensure no overlapping (too close) points among the geometries.
        Args:
            geometries (list[Geometry] | tuple[Geometry]): List of Geometry objects to concatenate.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Union {self.get_counter()}')
        if not geometries:
            return
        # Concatenate the x, y, z coordinates of all geometries
        self.xs = np.hstack([g.xs for g in geometries])
        self.ys = np.hstack([g.ys for g in geometries])
        self.zs = np.hstack([g.zs for g in geometries])


class Subtract(Geometry):
    """Pointwise subtraction: keep points in geo1 farther than rmax from any point in geo2."""

    def __init__(self, geo1: Geometry, geo2: Geometry, rmax: float = 1e-5, name=None):
        """
        Initialize a Subtract object that removes points in geo1 close to geo2.

        Args:
            geo1 (Geometry): The base geometry.
            geo2 (Geometry): The geometry to subtract from geo1.
            rmax (float, optional): Maximum distance for subtraction. Defaults to 1e-5.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Subtract {self.get_counter()}')
        if geo1.size == 0:
            return
        if geo2.size == 0:
            # If geo2 is empty, retain all points from geo1
            self.set_coord(geo1.xs, geo1.ys, geo1.zs)
            return
        # Build a KDTree for geo2 and query distances to geo1 points
        tree = KDTree(geo2.matrix_coords)
        d, _ = tree.query(geo1.matrix_coords)
        # Keep points in geo1 that are farther than rmax from geo2
        mask = d >= rmax
        self.set_coord(geo1.xs[mask], geo1.ys[mask], geo1.zs[mask])


class Intersect(Geometry):
    """Pointwise intersection of multiple geometries.

    Keeps points from the first geometry that are within `rmax` of at least one
    point in every other geometry (common intersection under tolerance).

    Usage:
    - Intersect(g1, g2, g3, ..., rmax=1e-5)
    - Intersect([g1, g2, g3, ...], rmax=1e-5)
    """

    def __init__(self, *geometries: Geometry | Iterable[Geometry], rmax: float = 1e-5, name=None):
        """
        Initialize an Intersect object that computes the intersection of multiple geometries.

        Args:
            geometries (Geometry | Iterable[Geometry]): Geometries to intersect.
            rmax (float, optional): Maximum distance for intersection. Defaults to 1e-5.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Intersect {self.get_counter()}')

        # Normalize input to a flat list of geometries
        if len(geometries) == 1 and isinstance(geometries[0], (list, tuple)):
            geos = list(geometries[0])
        else:
            geos = list(geometries)

        if len(geos) == 0:
            return
        if len(geos) == 1:
            # Degenerate case: intersection of one set is itself
            g0 = geos[0]
            if g0.size == 0:
                return
            self.set_coord(g0.xs, g0.ys, g0.zs)
            return

        if rmax < 0:
            raise ValueError('rmax must be non-negative')

        # If any geometry is empty, the intersection is empty
        if any(g.size == 0 for g in geos):
            return

        base = geos[0]
        base_coords = base.matrix_coords

        # Start with all base points and refine by intersecting with others
        mask = np.ones(base_coords.shape[0], dtype=bool)
        for g in geos[1:]:
            tree = KDTree(g.matrix_coords)
            d, _ = tree.query(base_coords)
            mask &= (d <= rmax)
            if not mask.any():
                break

        self.set_coord(base.xs[mask], base.ys[mask], base.zs[mask])


class Stack(Geometry):
    """Stack a 2D layer along an axis by repeating its points at dl-spacing."""

    def __init__(self, layer: Geometry, axis: str, n_axis: int, dl: float, name=None):
        """
        Initialize a Stack object that stacks a 2D layer along a specified axis.

        Args:
            layer (Geometry): The 2D layer to stack.
            axis (str): Axis along which to stack ('x', 'y', or 'z').
            n_axis (int): Number of repetitions along the axis.
            dl (float): Spacing between repetitions.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Stack {self.get_counter()}')
        if n_axis == 0 or layer.size == 0:
            return
        coord_layer = layer.matrix_coords
        axis2num = {'x': 0, 'y': 1, 'z': 2}
        a = axis2num[axis]
        level = coord_layer[0, a]
        if not np.allclose(coord_layer[:, a], level):
            raise ValueError('Layer must be planar orthogonal to the stacking axis')
        # Generate shifts along the stacking axis
        k = np.arange(0, n_axis, np.sign(n_axis), dtype=int)
        shifts = np.zeros((k.size, 3), dtype=float)
        shifts[:, a] = k * dl
        # Apply shifts to create stacked coordinates
        coords = (coord_layer[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        coords[:, a] += 0.0  # explicit
        self.set_coord(coords[:, 0], coords[:, 1], coords[:, 2])


class Clip(Geometry):
    """
    Half-space clipping by a named plane through the origin or an arbitrary plane.
    """

    def __init__(
        self,
        geo: Geometry,
        *,
        keep: str,
        plane_name: str | None = None,
        plane_normal: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        plane_point: list[float] | tuple[float, float, float] | np.ndarray | None = None,
        name=None,
    ):
        """
        Initialize a Clip object that clips a geometry by a plane.

        Rules:
            - If plane_name is given, plane_normal and plane_point must not be provided.
            - If plane_name is not given, plane_normal and plane_point must both be provided.

        Args:
            geo (Geometry): The source geometry to clip.
            keep (str): Side to keep ('positive' or 'negative').
            plane_name (str, optional): Named plane ('XOY', 'XOZ', 'YOZ'). Defaults to None.
            plane_normal (array-like, optional): Normal vector of the plane. Defaults to None.
            plane_point (array-like, optional): A point on the plane. Defaults to None.
            name (str, optional): Name of the resulting geometry. Defaults to None.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        super().__init__(name=name or f'Clip {self.get_counter()}')
        if geo.size == 0:
            return

        if keep not in ('positive', 'negative'):
            raise ValueError("keep must be 'positive' or 'negative'")

        coords = geo.matrix_coords

        # Determine plane definition
        if plane_name is not None:
            if plane_normal is not None or plane_point is not None:
                raise ValueError('Do not mix plane_name with plane_normal/plane_point')

            normals = {
                'XOY': np.array([0.0, 0.0, 1.0], dtype=float),  # z+
                'XOZ': np.array([0.0, 1.0, 0.0], dtype=float),  # y+
                'YOZ': np.array([1.0, 0.0, 0.0], dtype=float),  # x+
            }
            if plane_name not in normals:
                raise ValueError("plane_name must be one of {'XOY','XOZ','YOZ'}")

            n = normals[plane_name]
            p0 = np.zeros(3, dtype=float)

        else:
            if plane_normal is None or plane_point is None:
                raise ValueError('plane_normal and plane_point must be provided together when plane_name is not used')

            p0 = np.asarray(plane_point, dtype=float).reshape(3)
            n = np.asarray(plane_normal, dtype=float).reshape(3)
            norm = np.linalg.norm(n)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError('plane_normal must be a non-zero finite 3D vector')

        # Signed distance to plane
        s = (coords - p0) @ n

        # Keep side (inclusive boundary)
        if keep == 'positive':
            mask = s >= 0.0
        else:
            mask = s <= 0.0

        self.set_coord(geo.xs[mask], geo.ys[mask], geo.zs[mask])