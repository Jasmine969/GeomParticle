from __future__ import annotations
import numpy as np
from copy import deepcopy
from typing import Iterable, Tuple, List
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

    def __init__(self, name: str | None = None, dimension: int = None):
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

    def copy(self):
        """
        Create a deep copy of the geometry object.

        Returns:
            Geometry: A new geometry object with the same attributes.
        """
        g = self.__class__.__new__(self.__class__)
        g.__dict__.update(deepcopy(self.__dict__))
        return g

    def load_from(self, other: Geometry):
        """
        Copy coordinate arrays and dimension from another Geometry into this instance.

        Args:
            other (Geometry): Source geometry.

        Returns:
            Geometry: self for chaining.

        Raises:
            TypeError: If other is not a Geometry.
        """
        if not isinstance(other, Geometry):
            raise TypeError('other must be a Geometry')
        self.xs = other.xs.copy()
        self.ys = other.ys.copy()
        self.zs = other.zs.copy()
        self.dimension = other.dimension
        return self

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
        g = self.copy()
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
        g = self.copy()
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

        return Geometry(dimension=self.dimension
                        ).set_coord(rotated[:, 0], rotated[:, 1], rotated[:, 2])

    def union(self, geometries: Geometry | List[Geometry] | Tuple[Geometry], name: str | None = None):
        """
        Concatenate this geometry with others and return a new Geometry.

        Args:
            geometries (Geometry | List[Geometry] | Tuple[Geometry]): Other Geometry objects to union with.
            name (str | None): Optional name for the resulting geometry.

        Returns:
            Geometry: A new geometry object containing the union of points.
        """
        if isinstance(geometries, Geometry):
            geos = [geometries]
        else:
            geos = list(geometries)
        all_geos = [self] + geos
        # if no geometries (shouldn't happen since self exists), return empty
        if not any(g.size for g in all_geos):
            return Geometry(name=name)
        dimension = 2
        for g in all_geos:
            if g.dimension == 3:
                dimension = 3
                break
        xs = np.hstack([g.xs for g in all_geos if g.size > 0])
        ys = np.hstack([g.ys for g in all_geos if g.size > 0])
        zs = np.hstack([g.zs for g in all_geos if g.size > 0])
        return Geometry(name=name, dimension=dimension).set_coord(xs, ys, zs)

    def __add__(self, other):
        """
        Return the union of this geometry with another.

        Args:
            other (Geometry): Another geometry to union with.

        Returns:
            Geometry: A new geometry object after union.

        Raises:
            NotImplementedError: If other is not a Geometry instance.
        """
        if isinstance(other, Geometry):
            return self.union(other)
        return NotImplemented

    def __iadd__(self, other):
        """
        Return the union of this geometry with another in-place.

        Args:
            other (Geometry): Another geometry to union with.

        R
          Returns:
            Geometry: The updated geometry object (self) after the in-place union.

        Raises:
            NotImplementedError: If other is not a Geometry instance.
        """
        if isinstance(other, Geometry):
            self.load_from(self.union(other))
            return self
        return NotImplemented

    def subtract(self, geo2: Geometry, rmax: float = 1e-5, name: str | None = None):
        """
        Return points from self that are at least rmax away from any point in geo2.

        Args:
            geo2 (Geometry): Geometry to subtract from self.
            rmax (float): Minimum distance to consider a point as not overlapping.
            name (str | None): Optional name for the resulting geometry.

        Returns:
            Geometry: A new geometry object after subtraction.
        """
        if self.size == 0:
            return Geometry(name=name)
        if geo2.size == 0:
            return Geometry(name=name).set_coord(self.xs, self.ys, self.zs)
        tree = KDTree(geo2.matrix_coords)
        d, _ = tree.query(self.matrix_coords)
        mask = d >= rmax
        return Geometry(name=name, dimension=self.dimension
                        ).set_coord(self.xs[mask], self.ys[mask], self.zs[mask])

    def __sub__(self, other):
        if isinstance(other, Geometry):
            return self.subtract(other)
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, Geometry):
            self.load_from(self.subtract(other))
            return self
        return NotImplemented

    def intersect(self, geometries: Geometry | List[Geometry] | Tuple[Geometry],
                  rmax: float = 1e-5, name: str | None = None):
        """
        Keep points from self that are within rmax of at least one point in every other geometry.

        Args:
            geometries (Geometry | List[Geometry] | Tuple[Geometry]): Other Geometry objects to intersect with.
            rmax (float): Maximum distance to consider points as intersecting.
            name (str | None): Optional name for the resulting geometry.

        Returns:
            Geometry: A new geometry object after intersection.
        """
        others = list(geometries)
        if rmax < 0:
            raise ValueError('rmax must be non-negative')
        if any(g.size == 0 for g in [self] + others):
            return Geometry(name=name)
        mask = np.ones(self.size, dtype=bool)
        dimension = 3
        for g in [self] + others:
            if g.dimension == 2:
                dimension = 2
                break
        for g in others:
            tree = KDTree(g.matrix_coords)
            d, _ = tree.query(self.matrix_coords)
            mask &= (d <= rmax)
            if not mask.any():
                break
        return Geometry(name=name, dimension=dimension
                        ).set_coord(self.xs[mask], self.ys[mask], self.zs[mask])

    def stack(self, axis: str, n_axis: int, dl: float, dimension: int, name: str | None = None):
        """
        Stack a planar layer (self) along axis by repeating points at spacing dl.

        Args:
            axis (str): Axis to stack along ('x', 'y', or 'z').
            n_axis (int): Number of layers to stack. Positive for positive direction,
                          negative for negative direction.
            dl (float): Spacing between layers.
            dimension (int): Dimension of the resulting geometry (2 or 3).
            name (str | None): Optional name for the resulting geometry.

        Returns:
            Geometry: A new geometry object after stacking.
        """
        if n_axis == 0 or self.size == 0:
            return Geometry(name=name)
        coord_layer = self.matrix_coords
        axis2num = {'x': 0, 'y': 1, 'z': 2}
        if axis not in axis2num:
            raise ValueError("axis must be one of 'x','y','z'")
        a = axis2num[axis]
        level = coord_layer[0, a]
        if not np.allclose(coord_layer[:, a], level):
            raise ValueError('Layer must be planar orthogonal to the stacking axis')
        k = np.arange(0, n_axis, np.sign(n_axis), dtype=int)
        shifts = np.zeros((k.size, 3), dtype=float)
        shifts[:, a] = k * dl
        coords = (coord_layer[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        return Geometry(name=name, dimension=dimension
                        ).set_coord(coords[:, 0], coords[:, 1], coords[:, 2])

    def clip(
            self, keep: str, *,
            plane_name: str | None = None,
            plane_normal: Iterable[float] | np.ndarray | None = None,
            plane_point: Iterable[float] | np.ndarray | None = None,
            name: str | None = None,
    ):
        """
        Clip geometry by a half-space defined by a named plane or an arbitrary plane.

        Rules:
            - If plane_name is given, plane_normal and plane_point must not be provided.
            - If plane_name is not given, plane_normal and plane_point must both be provided.

        Args:
            keep (str): Side to keep ('positive' or 'negative').
            plane_name (str, optional): Named plane ('XOY', 'XOZ', 'YOZ'). Defaults to None.
            plane_normal (array-like, optional): Normal vector of the plane. Defaults to None.
            plane_point (array-like, optional): A point on the plane. Defaults to None.
            name (str, optional): Name of the resulting geometry. Defaults to None.

        Returns:
            Geometry: A new geometry object after clipping.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        if self.size == 0:
            return Geometry(name=name)
        if keep not in ('positive', 'negative'):
            raise ValueError("keep must be 'positive' or 'negative'")
        if plane_name is not None:
            if plane_normal is not None or plane_point is not None:
                raise ValueError('Do not mix plane_name with plane_normal/plane_point')
            normals = {
                'XOY': np.array([0.0, 0.0, 1.0], dtype=float),
                'XOZ': np.array([0.0, 1.0, 0.0], dtype=float),
                'YOZ': np.array([1.0, 0.0, 0.0], dtype=float),
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
        s = (self.matrix_coords - p0) @ n
        if keep == 'positive':
            mask = s >= 0.0
        else:
            mask = s <= 0.0
        return Geometry(name=name, dimension=self.dimension
                        ).set_coord(self.xs[mask], self.ys[mask], self.zs[mask])

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
        return Geometry(dimension=self.dimension).set_coord(x_del, y_del, z_del)

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

    def equal(self, geo: Geometry):
        """
        Check if two geometries are equal based on their coordinates.

        Args:
            geo (Geometry): Another geometry to compare with.

        Returns:
            bool: True if the geometries are equal, False otherwise.
        """
        return self.dimension == geo.dimension and np.array_equal(self.matrix_coords, geo.matrix_coords)

    def __eq__(self, other):
        """
        Check if two geometries are equal using the equal method.

        Args:
            other (Geometry): Another geometry to compare with.

        Returns:
            bool: True if the geometries are equal, False otherwise.

        Raises:
            NotImplementedError: If other is not a Geometry instance.
        """
        if isinstance(other, Geometry):
            return self.equal(other)
        return NotImplemented

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
