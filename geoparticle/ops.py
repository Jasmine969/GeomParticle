from __future__ import annotations
import numpy as np
from .base import Geometry


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
        if len(geometries) == 1:
            self.set_coord(geometries[0].xs, geometries[0].ys, geometries[0].zs)
            return
        base = geometries[0]
        me = base.union(geometries[1:])
        self.set_coord(me.xs, me.ys, me.zs)
        self.dimension = me.dimension
        self.check_overlap()


class Subtract(Geometry):
    """Pointwise subtraction: keep points in geo1 farther than `rmax` from any point in geo2."""

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
        me = geo1.subtract(geo2, rmax=rmax)
        self.set_coord(me.xs, me.ys, me.zs)
        self.check_overlap()


class Intersect(Geometry):
    """Pointwise intersection of multiple geometries.

    Keeps points from the first geometry that are within `rmax` of at least one
    point in every other geometry (common intersection under tolerance).

    Usage:
    - Intersect(g1, g2, g3, ..., rmax=1e-5)
    - Intersect([g1, g2, g3, ...], rmax=1e-5)
    """

    def __init__(self, geometries, rmax: float = 1e-5, name=None):
        """
        Initialize an Intersect object that computes the intersection of multiple geometries.

        Args:
            geometries (Tuple[Geometry] | List[Geometry]): Geometries to intersect.
            rmax (float, optional): Maximum distance for intersection. Defaults to 1e-5.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Intersect {self.get_counter()}')
        if len(geometries) == 1:
            self.set_coord(geometries[0].xs, geometries[0].ys, geometries[0].zs)
            return
        base = geometries[0]
        me = base.intersect(geometries[1:], rmax=rmax)
        self.set_coord(me.xs, me.ys, me.zs)
        self.check_overlap()


class Stack(Geometry):
    """Stack a 2D layer along an axis by repeating its points at dl-spacing."""

    def __init__(self, layer: Geometry, axis: str, n_axis: int,
                 dl: float, dimension: int, name=None):
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
        me = layer.stack(axis, n_axis, dl, dimension=dimension)
        self.set_coord(me.xs, me.ys, me.zs)
        self.check_overlap()


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
        me = geo.clip(
            keep=keep,
            plane_name=plane_name,
            plane_normal=plane_normal,
            plane_point=plane_point,
        )
        self.set_coord(me.xs, me.ys, me.zs)
        self.check_overlap()
