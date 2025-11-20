"""
This module provides various geometric operations on Geometry objects,
including shifting, mirroring, rotating, union, subtraction, intersection,
stacking, and clipping.
"""

from __future__ import annotations
import numpy as np
from .base import Geometry
from typing import Iterable, Tuple, List


class Shift(Geometry):
    """
    Shift geometry by given offsets in x, y, z directions.

    Shortest import: `from geoparticle import Shift`
    """

    def __init__(self, geo: Geometry,
                 x: float | None = None, y: float | None = None, z: float | None = None,
                 name=None):
        """
        Initialize a Shift object that shifts a geometry by specified offsets.

        Args:
            geo (Geometry): The source geometry to shift.
            x (float | None): Shift in the x-direction. Defaults to None.
            y (float | None): Shift in the y-direction. Defaults to None.
            z (float | None): Shift in the z-direction. Defaults to None.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Shift {self.get_counter()}', dimension=geo.dimension)
        self.load_from(geo.shift(x, y, z))


class Mirror(Geometry):
    """
    Mirror geometry across a specified plane.

    Shortest import: `from geoparticle import Mirror`
    """
    def __init__(self, geo: Geometry, plane_name: str, plane_pos: float, name=None):
        """
        Initialize a Mirror object that mirrors the geometry across a specified plane.

        Args:
            geo (Geometry): The source geometry to mirror.
            plane_name (str): Name of the plane ('YOZ', 'XOY', or 'XOZ').
            plane_pos (float): Position of the plane.

        Raises:
            ValueError: If an invalid plane name is provided.
        """
        super().__init__(name=name or f'Mirror {self.get_counter()}', dimension=geo.dimension)
        self.load_from(geo.mirror(plane_name, plane_pos))


class Rotate(Geometry):
    """
    Rotate geometry around an axis by a given angle.

    Shortest import: `from geoparticle import Rotate`
    """

    def __init__(self, geo: Geometry,
                 angle_deg: float, axis_direction: str | None = None,
                 axis_point1: Iterable[float] | None = None,
                 axis_point2: Iterable[float] | None = None,
                 name=None):
        """
        Initialize a Rotate object that rotates a geometry around a specified axis.

        Rules:
            - If axis_direction is provided, axis_point1 and axis_point2 must not be provided
            - If axis_direction is None, both axis_point1 and axis_point2 must be provided
        Args:
            geo (Geometry): The source geometry to rotate.
            angle_deg (float): Rotation angle in degrees.
            axis_direction (str | None): Principal axis ('x', 'y', or 'z').
            axis_point1 (Iterable[float] | None): First point defining the custom axis.
            axis_point2 (Iterable[float] | None): Second point defining the custom axis.

        Raises:
            ValueError: If invalid axis parameters are provided.
        """
        super().__init__(name=name or f'Rotate {self.get_counter()}', dimension=geo.dimension)
        self.load_from(geo.rotate(angle_deg, axis_direction, axis_point1, axis_point2))


class Union(Geometry):
    """
    Concatenate multiple geometries.

    Shortest import: `from geoparticle import Union`
    """

    def __init__(self, geometries: List[Geometry, ...] | Tuple[Geometry, ...], name=None):
        """
        Initialize a Union object that concatenates multiple geometries.
        Users had better ensure no overlapping (too close) points among the geometries.
        Args:
            geometries (list[Geometry, ...] | tuple[Geometry, ...]): List of Geometry objects to concatenate.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Union {self.get_counter()}')
        if len(geometries) == 1:
            self.set_coord(geometries[0].xs, geometries[0].ys, geometries[0].zs)
            return
        base = geometries[0]
        self.load_from(base.union(geometries[1:]))
        self.check_overlap()


class Subtract(Geometry):
    """
    Pointwise subtraction: keep points in geo1 farther than `rmax` from any point in geo2.

    Shortest import: `from geoparticle import Subtract`
    """

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
        self.load_from(geo1.subtract(geo2, rmax=rmax))
        self.check_overlap()


class Intersect(Geometry):
    """
    Pointwise intersection of multiple geometries.

    Keeps points from the first geometry that are within `rmax` of at least one
    point in every other geometry (common intersection under tolerance).

    Usage:
    - Intersect(g1, g2, g3, ..., rmax=1e-5)
    - Intersect([g1, g2, g3, ...], rmax=1e-5)

    Shortest import: `from geoparticle import Intersect`
    """

    def __init__(self, geometries: Tuple[Geometry, ...] | List[Geometry, ...],
                 rmax: float = 1e-5, name=None):
        """
        Initialize an Intersect object that computes the intersection of multiple geometries.

        Args:
            geometries (Tuple[Geometry, ...] | List[Geometry, ...]): Geometries to intersect.
            rmax (float, optional): Maximum distance for intersection. Defaults to 1e-5.
            name (str, optional): Name of the resulting geometry. Defaults to None.
        """
        super().__init__(name=name or f'Intersect {self.get_counter()}')
        if len(geometries) == 1:
            self.set_coord(geometries[0].xs, geometries[0].ys, geometries[0].zs)
            return
        base = geometries[0]
        self.load_from(base.intersect(geometries[1:], rmax=rmax))
        self.check_overlap()


class Stack(Geometry):
    """
    Stack a 2D layer along an axis by repeating its points at dl-spacing.

    Shortest import: `from geoparticle import Stack`
    """

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
        self.load_from(layer.stack(axis, n_axis, dl, dimension=dimension))
        self.check_overlap()


class Clip(Geometry):
    """
    Half-space clipping by a named plane through the origin or an arbitrary plane.

    Shortest import: `from geoparticle import Clip`
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
        self.load_from(geo.clip(
            keep=keep,
            plane_name=plane_name,
            plane_normal=plane_normal,
            plane_point=plane_point,
        ))
        self.check_overlap()
