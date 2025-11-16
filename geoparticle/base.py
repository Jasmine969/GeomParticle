from __future__ import annotations
import numpy as np
from copy import deepcopy
from typing import Iterable
from scipy.spatial import KDTree
from warnings import warn


class CounterMeta(type):
    """Metaclass that adds a class-wide counter for default naming of instances."""

    def __init__(cls, name, bases, attrs):
        """
        Initialize the metaclass and set the counter to 0.

        Args:
            name (str): Name of the class.
            bases (tuple): Base classes of the class.
            attrs (dict): Attributes of the class.
        """
        super().__init__(name, bases, attrs)
        cls.counter = 0


class Geometry(metaclass=CounterMeta):
    """
    Base class for geometry objects, providing utilities for vector transformations
    and coordinate management.
    """

    def __init__(self, name: str | None = None, dimension : int = None):
        """
        Initialize a Geometry object with optional naming.

        Args:
            name (str | None): Optional name for the geometry. Defaults to a
                               generated name based on the class name and counter.
        """
        type(self).counter += 1
        self.xs = np.array([], dtype=float)  # X-coordinates of the geometry
        self.ys = np.array([], dtype=float)  # Y-coordinates of the geometry
        self.zs = np.array([], dtype=float)  # Z-coordinates of the geometry
        self.name = name or f'{self.__class__.__name__} {self.get_counter()}'
        self.dimension = dimension


    def _copy(self):
        """
        Create a deep copy of the geometry object.

        Returns:
            Geometry: A new geometry object with the same attributes.
        """
        g = self.__class__.__new__(self.__class__)
        g.__dict__.update(deepcopy(self.__dict__))
        return g

    @classmethod
    def get_counter(cls) -> int:
        """
        Get the current value of the class-wide counter.

        Returns:
            int: The current counter value.
        """
        return cls.counter

    @property
    def size(self) -> int:
        """
        Get the number of points in the geometry.

        Returns:
            int: The size of the geometry.
        """
        return self.xs.size

    @property
    def matrix_coords(self) -> np.ndarray:
        """
        Get the coordinates as a 2D array.

        Returns:
            np.ndarray: Array of shape (N, 3) with [x, y, z] coordinates.
        """
        return np.c_[self.xs, self.ys, self.zs]

    @property
    def flatten_coords(self) -> np.ndarray:
        """
        Get the flattened array of coordinates.

        Returns:
            np.ndarray: Flattened array of [x, y, z] coordinates.
        """
        return self.matrix_coords.ravel()

    def set_coord(self, xs, ys, zs):
        """
        Set the coordinates of the geometry, broadcasting scalars if necessary.

        Args:
            xs: X-coordinates (scalar or array-like).
            ys: Y-coordinates (scalar or array-like).
            zs: Z-coordinates (scalar or array-like).

        Returns:
            Geometry: The updated geometry object.

        Raises:
            TypeError: If a scalar is provided without a size hint.
            ValueError: If the coordinate arrays have mismatched sizes.
        """

        def _to_array(v, size_hint=None):
            """
            Convert a value to a numpy array, broadcasting scalars if needed.

            Args:
                v: Value to convert.
                size_hint (int | None): Size hint for broadcasting scalars.

            Returns:
                np.ndarray: Converted array.

            Raises:
                TypeError: If a scalar is provided without a size hint.
            """
            if np.isscalar(v):
                if size_hint is None:
                    raise TypeError('Cannot broadcast scalar without size hint')
                return np.full(size_hint, v, dtype=float)
            arr = np.asarray(v, dtype=float).ravel()
            return arr

        # Derive size from the first array-like input
        size = None
        for v in (xs, ys, zs):
            if not np.isscalar(v):
                size = np.asarray(v).size
                break

        xs = _to_array(xs, size)
        ys = _to_array(ys, size)
        zs = _to_array(zs, size)
        if not (xs.size == ys.size == zs.size):
            raise ValueError('Coordinate arrays must have the same size')
        self.xs, self.ys, self.zs = xs.copy(), ys.copy(), zs.copy()
        return self

    def shift(self, x=0.0, y=0.0, z=0.0):
        """
        Translate the geometry by the given offsets.

        Args:
            x (float): Offset along the X-axis.
            y (float): Offset along the Y-axis.
            z (float): Offset along the Z-axis.

        Returns:
            Geometry: A new geometry object with shifted coordinates.
        """
        g = self._copy()
        g.xs = g.xs + x
        g.ys = g.ys + y
        g.zs = g.zs + z
        return g

    def mirror(self, plane_name: str, plane_pos: float):
        """
        Mirror the geometry across a principal plane.

        Args:
            plane_name (str): Name of the plane ('YOZ', 'XOY', or 'XOZ').
            plane_pos (float): Position of the plane.

        Returns:
            Geometry: A new geometry object mirrored across the specified plane.

        Raises:
            ValueError: If an invalid plane name is provided.
        """
        g = self._copy()
        if plane_name == 'YOZ':
            g.xs = plane_pos * 2 - g.xs
        elif plane_name == 'XOY':
            g.zs = plane_pos * 2 - g.zs
        elif plane_name == 'XOZ':
            g.ys = plane_pos * 2 - g.ys
        else:
            raise ValueError('Invalid plane_name')
        return g

    def rotate(self, angle_deg: float, axis_direction: str | None = None,
               axis_point1: Iterable[float] | None = None,
               axis_point2: Iterable[float] | None = None):
        """
        Rotate the geometry around a principal axis or a custom axis.

        Rules:
            - If axis_direction is provided, axis_point1 and axis_point2 must not be provided
            - If axis_direction is None, both axis_point1 and axis_point2 must be provided
        Args:
            angle_deg (float): Rotation angle in degrees.
            axis_direction (str | None): Principal axis ('x', 'y', or 'z').
            axis_point1 (Iterable[float] | None): First point defining the custom axis.
            axis_point2 (Iterable[float] | None): Second point defining the custom axis.

        Returns:
            Geometry: A new geometry object with rotated coordinates.

        Raises:
            ValueError: If invalid axis parameters are provided.
        """
        from scipy.spatial.transform import Rotation
        angle = np.deg2rad(angle_deg)
        if axis_direction is not None:
            if axis_point2 is not None:
                raise ValueError('Provide at most axis_point1 when axis_direction is set')
            p1 = np.array(axis_point1 if axis_point1 is not None else (0, 0, 0), dtype=float)
            if axis_direction == 'x':
                axis_vec = np.array([1.0, 0.0, 0.0])
            elif axis_direction == 'y':
                axis_vec = np.array([0.0, 1.0, 0.0])
            elif axis_direction == 'z':
                axis_vec = np.array([0.0, 0.0, 1.0])
            else:
                raise ValueError("axis_direction must be 'x','y' or 'z'")
        else:
            if axis_point1 is None or axis_point2 is None:
                raise ValueError('Provide axis_point1 and axis_point2 when axis_direction is None')
            p1 = np.asarray(axis_point1, dtype=float)
            p2 = np.asarray(axis_point2, dtype=float)
            axis_vec = p2 - p1
            nrm = np.linalg.norm(axis_vec)
            if nrm == 0:
                raise ValueError('Axis points must not coincide')
            axis_vec = axis_vec / nrm

        rot = Rotation.from_rotvec(axis_vec * angle)
        pts = np.column_stack((self.xs, self.ys, self.zs))
        rotated = rot.apply(pts - p1) + p1

        g = self._copy()
        g.set_coord(rotated[:, 0], rotated[:, 1], rotated[:, 2])
        return g

    def get_and_delete(self, ids: np.ndarray):
        """
        Extract points by their indices and remove them from the geometry.

        Args:
            ids (np.ndarray): Indices of points to extract.

        Returns:
            Geometry: A new geometry object containing the extracted points.
        """
        ids = np.asarray(ids, dtype=int)
        mask = np.zeros(self.size, dtype=bool)
        mask[ids] = True
        x_del, self.xs = self.xs[ids], self.xs[~mask]
        y_del, self.ys = self.ys[ids], self.ys[~mask]
        z_del, self.zs = self.zs[ids], self.zs[~mask]
        return Geometry().set_coord(x_del, y_del, z_del)

    def coord2id(self, x: float, y: float, z: float):
        """
        Find the nearest vertex index/indices and its/their coordinates.

        Args:
            x (float): X-coordinate of the query point.
            y (float): Y-coordinate of the query point.
            z (float): Z-coordinate of the query point.

        Returns:
            tuple: Index/Indices of the nearest vertex/vertices and its/their coordinates.
        """
        tree = KDTree(self.matrix_coords)
        r_min, idx = tree.query([x, y, z])
        indices_all = tree.query_ball_point([x, y, z], r=r_min + 1e-10)
        return indices_all, self.matrix_coords[indices_all]

    def check_overlap(self, tol: float = 1e-10):
        """
        Check if there are overlapping points in the geometry.
        Find two nearest points and see if their distance is less than tol.

        Args:
            tol (float): Tolerance distance to consider points as overlapping.
        """
        if self.size == 1:
            return
        tree = KDTree(self.matrix_coords)
        dists, indices = tree.query(self.matrix_coords, k=[2])
        min_idx = np.argmin(dists)
        min_pair_idx = indices[min_idx].item()
        min_distance = dists[min_idx].item()
        point1 = self.matrix_coords[min_idx]
        point2 = self.matrix_coords[min_pair_idx].round(4)
        if min_distance < tol:
            warn(f'Overlap detected between point #{min_idx} '
                 f'{point1[0].item(), point1[1].item(), point1[2].item()}'
                 f' and #{min_pair_idx} {point2[0].item(), point2[1].item(), point2[2].item()} '
                 f'with a distance of {min_distance: .3g} < tol={tol:.3g}')


    def plot(self, ax=None, ms=None, alpha=None, **scatter_kwargs):
        """
        Plot the geometry points in 2D.

        Args:
            ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
            **scatter_kwargs: Additional keyword arguments for the scatter plot.
            ms: Marker size.
            alpha: Transparency level of the points.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            external_flag = False
            if self.dimension == 2:
                ax = plt.axes()
            else:
                ax = plt.axes(projection='3d')
        else:
            external_flag = True
        if self.dimension == 2:
            alpha = 1 if alpha is None else alpha
            ms = 25 if ms is None else ms
            ax.scatter(self.xs, self.ys, s=ms, alpha=alpha, **scatter_kwargs)
        else:
            alpha = 0.5 if alpha is None else alpha
            ms = 10 if ms is None else ms
            ax.scatter(self.xs, self.ys, self.zs, s=ms, alpha=alpha, **scatter_kwargs)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if self.dimension != 2:
            ax.set_zlabel('z')
        if external_flag:
            return ax
        ax.axis('equal')
        plt.show()
        return None
