"""
Miscellaneous geometries made up of particles with a spacing of dl,
including 2D geometries (thick rectangle, filled rectangle, thick/filled ring),
3D geometries (block, side surface of a cylinder, torus).
Diverse transformations are also provided, including
translation, mirror, rotation, stack, and union.
All the geometry are first created in the default coordinate system
along the y-axis (3D solids of revolution) or in the XOZ plane (2D),
then obtain their required coordinates by transform_coordinate.
The default coordinate system is x (inward) y (up) z (right)
    y
    | x
    |/
    ————z
Some utility functions are also provided in the beginning.
"""
import numpy as np
from warnings import warn


# Utility functions ===========================
def get_n_per_ring(r, d, phi_ring=2 * np.pi):
    """
    Count the particle of a ring by the radius and spacing.
    :param r: radius
    :param d: spacing
    :param phi_ring: radian
    :return: particle count
    """
    n = np.ones_like(r, dtype=float).flatten()
    r = np.asarray(r, dtype=float).flatten()
    n[r != 0] = phi_ring / np.arccos(1 - 0.5 * (d / r[r != 0]) ** 2)
    if n.size == 1:
        return round(n.item())
    return np.round(n).astype(int)


def get_spacing_ring(r, n, phi_ring=2 * np.pi):
    """
    Calculate the spacing of neighboring particles of a ring
    by the radius and particle count.
    :param r: radius
    :param n: particle count
    :param phi_ring: radian
    :return: spacing
    """
    d = r * np.sqrt(2 * (1 - np.cos(phi_ring / n)))
    return d


def cart_coord_ring(n, r):
    """
    Get cartesian coordinates of particles of a ring.
    :param n: particle count of the ring
    :param r: radius
    :return: abscissa and ordinate
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x1, x2 = r * np.cos(angles), r * np.sin(angles)
    return x1, x2


def transform_coordinate(xs, ys, zs, **kwargs):
    """
    :param xs: self-explanatory
    :param ys: self-explanatory
    :param zs: self-explanatory
    :param kwargs: axis or plane, the axis is the normal of the plane
    :return: xs, ys, zs in the new coordinate system
    """
    plane2axis = {'XOY': 'z', 'YOZ': 'x', 'XOZ': 'y'}
    assert len(kwargs) == 1
    if 'axis' in kwargs:
        axis_up = kwargs['axis']
        assert axis_up in ['x', 'y', 'z']
    elif 'plane' in kwargs:
        plane = kwargs['plane']
        assert plane in plane2axis.keys()
        axis_up = plane2axis[plane]
    else:
        raise KeyError('Kwargs can only be axis or plane!')
    if axis_up == 'z':
        xs, ys, zs = zs, xs, ys
    elif axis_up == 'x':
        xs, ys, zs = ys, zs, xs
    return xs, ys, zs


def get_wall_ID(i, j, n_per_ring, smallest_ID=1):
    """
    get the ID of cylinder wall
    :param i: ID on the ring
    :param j: ID on the axis
    :param n_per_ring: self-explanatory
    :param smallest_ID: self-explanatory
    :return: wall-ID
    """
    if i > n_per_ring:
        i = i % n_per_ring
    return (j - 1) * n_per_ring + i + smallest_ID - 1


# Geometries ================================
class CounterMeta(type):
    """MetaClass, create a counter for the base class"""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.counter = 0


class Geometry(metaclass=CounterMeta):
    """Base class"""

    def __init__(self, name=None):
        type(self).counter += 1
        self.xs = np.array([])
        self.ys = np.array([])
        self.zs = np.array([])
        self.log = ''
        self.l_axis = 0
        self.n_axis = 0
        self.n_ring = 0
        self.name = f'Geometry {self.get_counter()}' if name is None else name

    @classmethod
    def get_counter(cls):
        return cls.counter

    @property
    def size(self):
        return self.xs.size

    @property
    def flatten_coords(self):
        return np.c_[self.xs, self.ys, self.zs].flatten()

    @property
    def matrix_coords(self):
        return np.c_[self.xs, self.ys, self.zs]

    def set_coord(self, xs, ys, zs):
        if isinstance(xs, np.ndarray):
            size = xs.size
        elif isinstance(ys, np.ndarray):
            size = ys.size
        else:
            raise TypeError('At least two coordinates should be ndarray')
        if isinstance(xs, int):
            xs = np.full(size, xs)
        elif isinstance(ys, int):
            ys = np.full(size, ys)
        elif isinstance(zs, int):
            zs = np.full(size, zs)
        self.xs = xs.copy()
        self.ys = ys.copy()
        self.zs = zs.copy()
        return self

    def shift(self, x=0, y=0, z=0):
        """Translation"""
        new_geo = self._copy()
        new_geo.xs += x
        new_geo.ys += y
        new_geo.zs += z
        return new_geo

    def mirror(self, plane_name, plane_pos):
        new_geo = self._copy()
        if plane_name == 'YOZ':
            new_geo.xs = plane_pos * 2 - self.xs
        elif plane_name == 'XOY':
            new_geo.zs = plane_pos * 2 - self.zs
        elif plane_name == 'XOZ':
            new_geo.ys = plane_pos * 2 - self.ys
        else:
            raise ValueError('Invalid plane_name!')
        return new_geo

    def rotate(self, angle, axis_direction=None, axis_point1=None, axis_point2=None):
        """
        :param angle: float. degree unit
        :param axis_direction: str. direction of rotational axis, can be x/y/z
        :param axis_point1: the first point on the axis (default: (0,0,0) if axis_direction is specified)
        :param axis_point2: the second point on the axis (only used when axis_direction is not specified)
        Users can determine the axis
        1. by direction and a point
            rotate(90, axis='x', axis_point1=(0,0,0))
        2. by two points
            rotate(90, axis_point1=(0,0,0), axis_point2=(0,0,1))
        """
        from scipy.spatial.transform import Rotation

        angle = np.deg2rad(angle)
        # determine the rotational axis
        if axis_direction is not None:
            if axis_point2 is not None:
                raise ValueError('Only axis_point1 can be specified when axis_direction is specified!')
            if axis_point1 is None:
                axis_point1 = np.array([0, 0, 0])
            else:
                axis_point1 = np.array(axis_point1)

            if axis_direction == 'x':
                axis_vector = np.array([1, 0, 0])
            elif axis_direction == 'y':
                axis_vector = np.array([0, 1, 0])
            elif axis_direction == 'z':
                axis_vector = np.array([0, 0, 1])
            else:
                raise ValueError("axis_direction must be 'x', 'y', or 'z'")
        elif axis_point1 is not None and axis_point2 is not None:
            axis_vector = np.array(axis_point2) - np.array(axis_point1)
            axis_vector = axis_vector / np.linalg.norm(axis_vector)  # normalization
        else:
            raise ValueError("Must specify either axis or both axis_point1 and axis_point2")

        rotation = Rotation.from_rotvec(axis_vector * angle)
        points = np.column_stack((self.xs, self.ys, self.zs))
        points_centered = points - axis_point1
        rotated_points = rotation.apply(points_centered) + axis_point1

        new_geo = self._copy()
        new_geo.set_coord(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2])
        return new_geo

    def get_and_delete(self, ids):
        mask = np.zeros(self.size).astype(bool)
        mask[ids] = True
        # geo_del should have the same order as ids
        # so it must be obtained by self.xs[ids] instead of self.xs[mask]
        x_del, self.xs = self.xs[ids], self.xs[~mask]
        y_del, self.ys = self.ys[ids], self.ys[~mask]
        z_del, self.zs = self.zs[ids], self.zs[~mask]
        geo_del = Geometry().set_coord(x_del, y_del, z_del)
        return geo_del

    def coord2id(self, x, y, z):
        from scipy.spatial import KDTree

        tree = KDTree(np.c_[self.xs, self.ys, self.zs])
        _, idx = tree.query([x, y, z])
        return idx, (self.xs[idx].item(), self.ys[idx].item(), self.zs[idx].item())

    def _copy(self):
        from copy import deepcopy
        new_geo = self.__class__.__new__(self.__class__)
        new_geo.__dict__.update(deepcopy(self.__dict__))
        return new_geo


# Create geometries by transformations ===========================
class Union(Geometry):
    def __init__(self, geometries, name=None):
        super().__init__(name=f'Union {self.get_counter()}' if name is None else name)
        self.xs = np.hstack(tuple(geometry.xs for geometry in geometries))
        self.ys = np.hstack(tuple(geometry.ys for geometry in geometries))
        self.zs = np.hstack(tuple(geometry.zs for geometry in geometries))


class Subtract(Geometry):
    """Subtraction operation: geo1 - geo2"""

    def __init__(self, geo1, geo2, rmin=1e-5, name=None):
        """
        Calculate the difference between geo1 and geo2 (geo1 - geo2)

        :param geo1: first geometry (minuend)
        :param geo2: second geometry (subtrahend)
        :param rmin: tolerance distance, points closer than rmin are considered identical
        :param name: name of the resulting geometry
        """
        super().__init__(name=f'Subtract {self.get_counter()}' if name is None else name)

        # Handle empty geometry cases
        if geo1.size == 0:
            self.xs = np.array([])
            self.ys = np.array([])
            self.zs = np.array([])
            return
        if geo2.size == 0:
            self.xs = geo1.xs.copy()
            self.ys = geo1.ys.copy()
            self.zs = geo1.zs.copy()
            return

        # Build KDTree for geo2 for efficient nearest neighbor search
        from scipy.spatial import KDTree
        geo2_points = np.column_stack((geo2.xs, geo2.ys, geo2.zs))
        tree = KDTree(geo2_points)

        # Query nearest distances from geo1 points to geo2
        geo1_points = np.column_stack((geo1.xs, geo1.ys, geo1.zs))
        distances, _ = tree.query(geo1_points)

        # Keep points that are not in geo2 (distance >= rmin)
        mask = distances >= rmin
        self.xs = geo1.xs[mask]
        self.ys = geo1.ys[mask]
        self.zs = geo1.zs[mask]


class Stack(Geometry):
    """Stack a 2D geometry along an axis
    If n_axis<0, stack along the negative axis
    """

    def __init__(self, layer, axis, n_axis, dl, name=None):
        # if n_axis_si>0, along the positive axis; else if n_axis_si<0, along the negative axis
        super().__init__(name=f'Stack {self.get_counter()}' if name is None else name)
        coord_layer = np.c_[layer.xs, layer.ys, layer.zs]
        axis2num = {'x': 0, 'y': 1, 'z': 2}
        level = coord_layer[0, axis2num[axis]]
        assert np.all(coord_layer[:, axis2num[axis]] == level)  # ensure the layer is 2D
        coords = np.zeros((np.abs(n_axis) * layer.xs.size, 3))
        for cur_axis, col in axis2num.items():
            if cur_axis == axis:
                coords[:, col] = np.repeat(np.arange(0, n_axis, np.sign(n_axis)) * dl, layer.xs.size) + level
            else:
                coords[:, col] = np.tile(coord_layer[:, col], np.abs(n_axis))
        self.xs = coords[:, 0]
        self.ys = coords[:, 1]
        self.zs = coords[:, 2]


# For 2D simulation ==========================
class Line(Geometry):
    def __init__(self, length, direction, dl, name=None):
        super().__init__(
            name=f'Line {self.get_counter()}' if name is None else name)
        ys = np.arange(0, length + dl * 0.1, dl)
        self.length = ys[-1]
        if self.length != length:
            warn('The length has been modified! Be careful to shift.')
        self.log += f'{self.name}: real/ori length is {self.length}/{length}\n'
        xs = np.zeros_like(ys)
        zs = xs.copy()
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=direction)


class SymmLines(Geometry):
    def __init__(self, length, direction, dist_half, dl, name=None):
        """
        Two symmetric lines along a direction, centered at origin.
        :param length: self-explanatory
        :param direction: x/y/z
        :param dist_half: distance of one line to the symmetry line
        :param dl: self-explanatory
        :param name: self-explanatory
        """
        super().__init__(
            name=f'Line {self.get_counter()}' if name is None else name)
        line_upper = Line(length, 'z', dl).shift(x=dist_half)
        line_lower = line_upper.mirror('YOZ', 0)
        me = Union((line_upper, line_lower))
        self.xs, self.ys, self.zs = transform_coordinate(me.xs, me.ys, me.zs, axis=direction)


class Arc(Geometry):
    def __init__(self, r, phi_range, plane_pos, plane, dl, name=None):
        super().__init__(name=f'Arc {self.get_counter()}' if name is None else name)
        phi_min, phi_max = (float(each) for each in phi_range[1:-1].split(','))
        phi_tot = phi_max - phi_min
        self.phi_tot = phi_tot
        assert phi_tot <= 360
        if phi_tot == 360:
            incl_min, incl_max = True, False
        else:
            if phi_range[0] == '[':
                incl_min = True
            elif phi_range[0] == '(':
                incl_min = False
            else:
                raise ValueError('Include sign can only be [ or (')
            if phi_range[-1] == ']':
                incl_max = True
            elif phi_range[-1] == ')':
                incl_max = False
            else:
                raise ValueError('Include sign can only be [ or (')
        phi_min = np.deg2rad(phi_min)
        phi_max = np.deg2rad(phi_max)
        phi_tot = phi_max - phi_min
        n_phi = get_n_per_ring(r, dl, phi_tot)
        d_phi = phi_tot / n_phi
        phis = d_phi * np.arange(int(not incl_min), n_phi + int(incl_max)) + phi_min
        zs, xs = r * np.cos(phis), r * np.sin(phis)
        ys = np.full_like(xs, plane_pos)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, plane=plane)


class Circle(Arc):
    def __init__(self, r, plane_pos, plane, dl, name=None):
        super().__init__(
            r, '[0,360)', plane_pos, plane, dl,
            name=f'Circle {self.get_counter()}' if name is None else name)


class Torus2D(Geometry):
    def __init__(self, r_ring, r_t, dl, plane='XOZ', phi_range='[0, 360)', name=None):
        super().__init__(name=f'Torus2D {self.get_counter()}' if name is None else name)
        arc_inner = Arc(r_t - r_ring, phi_range, 0, 'XOZ', dl)
        arc_outer = Arc(r_t + r_ring, phi_range, 0, 'XOZ', dl)
        me = Union((arc_inner, arc_outer))
        self.xs, self.ys, self.zs = transform_coordinate(me.xs, me.ys, me.zs, plane=plane)


class Rectangle(Geometry):
    """2D Rectangle"""

    def __init__(self, length, width, plane_pos, axis, dl,
                 name=None):
        super(Rectangle, self).__init__(
            name=f'Rectangle {self.get_counter()}' if name is None else name
        )
        z_bot = np.arange(0, length + dl * 0.1, dl).round(6)
        self.length = z_bot[-1]
        if self.length != length:
            warn('The length has been modified! Be careful to shift.')
        self.log += f'{self.name}: real/ori length is {self.length}/{length}\n'
        x_left = np.arange(dl, width + dl * 0.1, dl).round(6)
        self.width = x_left[-1]
        if self.width != width:
            warn('The width has been modified! Be careful to shift.')
        self.log += f'real/ori width is {self.width}/{width}'
        x_left = x_left[:-1]
        z_left = np.full_like(x_left, 0)
        x_right = np.copy(x_left)
        z_right = np.full_like(x_right, self.length)
        x_bot = np.full_like(z_bot, 0)
        z_top = np.copy(z_bot)
        x_top = np.full_like(z_top, self.width)
        zs = np.r_[z_bot, z_left, z_top, z_right]
        xs = np.r_[x_bot, x_left, x_top, x_right]
        ys = np.full_like(xs, plane_pos)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)
        print(self.log)


class ThickRectangle(Geometry):
    """2D FilledRectangle with thickness.
    Sometimes the wall has to be thickened to avoid penetration.
    Length and width are the inner dimensions.
    """

    def __init__(self, length, width, n_thick, plane_pos, axis, dl,
                 name=None):
        super(ThickRectangle, self).__init__(
            name=f'ThickRectangle {self.get_counter()}' if name is None else name
        )
        gms = []
        for i_layer in range(n_thick):
            cur_length = length + i_layer * 2 * dl
            cur_width = width + i_layer * 2 * dl
            layer = Rectangle(cur_length, cur_width, plane_pos, 'y', dl
                              ).shift(x=-i_layer * dl, z=-i_layer * dl)
            gms.append(layer)
        me = Union(gms)
        self.xs, self.ys, self.zs = transform_coordinate(me.xs, me.ys, me.zs, axis=axis)


class FilledRectangle(Geometry):
    """2D Filled rectangle"""

    def __init__(self, length, width, plane_pos, axis, dl, name=None):
        super().__init__(name=f'FilledRectangle {self.get_counter()}' if name is None else name)
        z = np.arange(0, length + dl * 0.1, dl).round(6)
        self.length = z[-1]
        if self.length != length:
            warn('The length has been modified! Be careful to shift.')
        self.log += f'{self.name}: real/ori length is {self.length}/{length}\n'
        x = np.arange(0, width + dl * 0.1, dl).round(6)
        self.width = x[-1]
        if self.width != width:
            warn('The width has been modified! Be careful to shift.')
        self.log += f'real/ori width is {self.width}/{width}'
        zs, xs = np.meshgrid(z, x)
        zs, xs = zs.flatten(), xs.flatten()
        ys = np.full_like(xs, plane_pos)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)
        print(self.log)


class ThickRing(Geometry):
    def __init__(
            self, r_out, r_in, dl, incl_inner, incl_outer, axis='y',
            adjust_dl=False, equal_size_per_circle=False,
            name=None
    ):
        """
        2D ring with thickness r_out-r_in, i.e.,
        the region between two concentric circles with radii of r_out and r_in.
        If r_in = 0, it is a filled circle.
        Multiple thick rings can be stacked to create a region between two coaxial cylinders.
        :param r_out: outer radius
        :param r_in: inner radius
        :param dl: spacing
        :param incl_inner: whether to include the inner ring
        :param incl_outer: whether to include the outer ring
        :param axis: axis of the ring. if axis is y, it is on the XOZ plane
        :param adjust_dl: whether to make the global dl = inner dl
        :param equal_size_per_circle: whether each ring has the same number of particles as the inner ring
        """
        super().__init__(name=f'ThickRing {self.get_counter()}' if name is None else name)
        self.dl = dl
        self.n_ring_in = get_n_per_ring(r_in, self.dl)
        if adjust_dl:
            assert r_in > 0
            self.dl = get_spacing_ring(r_in, self.n_ring_in)
        n_radial = round((r_out - r_in) / self.dl)
        self.r_out = n_radial * self.dl + r_in
        self.log = (f'{self.name}: real/ori r_out {self.r_out}/{r_out},'
                    f' real/ori dl {self.dl}/{dl}')
        rs = np.arange(0, n_radial + 1) * self.dl + r_in
        if equal_size_per_circle:
            n_per_rings = np.full_like(rs, self.n_ring_in).astype(int)
        else:
            n_per_rings = get_n_per_ring(rs, self.dl)
            assert self.n_ring_in == n_per_rings[0]
        if not incl_inner:
            rs = rs[1:]
            n_per_rings = n_per_rings[1:]
        if not incl_outer:
            rs = rs[:-1]
            n_per_rings = n_per_rings[:-1]
        self.rs = rs
        self.n_per_rings = n_per_rings
        zs, xs = [], []
        for r, n in zip(self.rs, self.n_per_rings):
            z, x = cart_coord_ring(n, r)
            zs.extend(z)
            xs.extend(x)
        zs, xs = np.asarray(zs), np.asarray(xs)
        ys = np.full_like(zs, 0)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)
        print(self.log)


class FilledCircle(ThickRing):
    def __init__(
            self, r, dl, axis='y',
            equal_size_per_circle=False,
            name=None
    ):
        """
        2D filled circle
        :param r: radius
        :param dl: spacing
        :param axis: axis of the circle. if axis is y, it is on the XOZ plane
        :param equal_size_per_circle: whether each ring has the same number of particles as the inner ring
        """
        super().__init__(
            r_out=r, r_in=0, dl=dl,
            incl_outer=True, incl_inner=True,
            axis=axis, equal_size_per_circle=equal_size_per_circle,
            name=f'FilledCircle {self.get_counter()}' if name is None else name
        )


# 3D Geometries ==================================
class Block(Geometry):
    def __init__(self, length, width, height, dl, name=None):
        """
        :param length: dx
        :param width: dy
        :param height: dz
        """
        super().__init__(name=f'Block {self.get_counter()}' if name is None else name)
        layer = FilledRectangle(length, width, 0, 'z', dl)
        n_height = int(height / dl) + 1
        self.height = (n_height - 1) * dl
        self.log = f'{self.name}: real/ori height {self.height}/{height}'
        me = Stack(layer, 'z', n_height, dl)
        self.set_coord(me.xs, me.ys, me.zs)
        print(self.log)


class ThickBlockWall(Geometry):
    def __init__(self, length, width, height, n_thick, dl, name=None):
        super().__init__(name=f'ThickBlockWall {self.get_counter()}' if name is None else name)
        # create the side wall (along the z-axis)
        layer = ThickRectangle(length, width, n_thick, 0, 'z', dl)
        n_height = int(height / dl) + 1
        side = Stack(layer, 'z', n_height, dl)
        # create the lower and upper lids
        lid_lower = Stack(
            FilledRectangle(length - 2 * dl, width - 2 * dl, 0, 'z', dl).shift(x=dl, y=dl),
            'z', -n_thick, dl)
        z_mid = (n_height - 1) / 2 * dl
        lid_upper = lid_lower.mirror('XOY', z_mid)
        me = Union((side, lid_lower, lid_upper))
        self.set_coord(me.xs, me.ys, me.zs)


class CylinderSide(Geometry):
    """3D Side surface of a cylinder"""

    def __init__(
            self, r, l_axis, dl, axis='y', name=None
    ):
        super().__init__(name=f'CylinderSide {self.get_counter()}' if name is None else name)
        self.l_axis = l_axis
        self.r = r
        self.n_axis = int(self.l_axis / dl) + 1
        # use y-axis as the cylinder-axis
        y = np.arange(0, self.n_axis) * dl
        self.l_axis = y[-1]
        self.log = f'{self.name}: real/ori length is {self.l_axis}/{l_axis}'
        if self.l_axis != l_axis:
            warn('The length has been modified! Be careful to shift.')
        self.n_ring = get_n_per_ring(r, dl)
        z, x = cart_coord_ring(self.n_ring, r)
        zs = np.tile(z, self.n_axis)
        xs = np.tile(x, self.n_axis)
        ys = np.repeat(y, self.n_ring)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)
        print(self.log)

    @property
    def dl_in_ring(self):
        """dl in a ring can be a little different from the specified dl"""
        return get_spacing_ring(self.r, self.n_ring)

    def get_and_delete_cy(self, ind: int, direction: str):
        """
        Get layers of atom coordinate and delete these atoms.
        :param ind: int, e.g., {0}, [n_axis_si-1, n_axis_si].
        :param direction: str, 'smaller' will get & delete [0,ind),
        'bigger' will get & delete (ind, n_axis_si].
        :return: xs, ys, zs
        """
        coords = np.c_[self.xs, self.ys, self.zs].reshape((-1, self.n_ring, 3))
        ind_all = list(range(self.n_axis))
        if direction == 'smaller':
            ind_del = ind_all[:ind]
            ind_remain = ind_all[ind:]
        elif direction == 'larger':
            ind_del = ind_all[ind + 1:]
            ind_remain = ind_all[:ind + 1]
        else:
            raise ValueError('Direction can only be smaller or larger')
        x_del = coords[ind_del, :, 0].flatten()
        y_del = coords[ind_del, :, 1].flatten()
        z_del = coords[ind_del, :, 2].flatten()
        self.xs = coords[ind_remain, :, 0].flatten()
        self.ys = coords[ind_remain, :, 1].flatten()
        self.zs = coords[ind_remain, :, 2].flatten()
        self.n_axis = len(ind_remain)
        geo_del = Geometry().set_coord(x_del, y_del, z_del)
        return geo_del


class Torus(Geometry):
    def __init__(
            self, r_ring, r_t, dl, n_ring,
            plane='XOZ',
            phi_range='[180,270)',
            regular_id=False,
            name=None,
    ):
        """
        3D Torus surface
        :param r_ring: radius of the smaller ring (section)
        :param r_t: radius of the larger circle
        :param dl: spacing
        :param n_ring: particle count per ring
        :param plane: XOY/YOZ/XOZ
        :param phi_range: radian range of the larger circle.
        :param regular_id: each larger ring has equal particles
            so that users can easily divide the torus atoms into groups (used in duodenum)
            Use interval notation: [/] is including and (/) is not
        """
        super().__init__(name=f'Torus {self.get_counter()}' if name is None else name)
        assert r_ring < r_t
        self.n_theta = n_ring
        self.n_phi = None
        phi_min, phi_max = (float(each) for each in phi_range[1:-1].split(','))
        phi_tot = phi_max - phi_min
        self.phi_tot = phi_tot
        assert phi_tot <= 360
        if phi_tot == 360:
            incl_min, incl_max = True, False
        else:
            if phi_range[0] == '[':
                incl_min = True
            elif phi_range[0] == '(':
                incl_min = False
            else:
                raise ValueError('Include sign can only be [ or (')
            if phi_range[-1] == ']':
                incl_max = True
            elif phi_range[-1] == ')':
                incl_max = False
            else:
                raise ValueError('Include sign can only be [ or (')
        self.atom_id_boundary = None

        thetas = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        r_P = r_t - r_ring * np.cos(thetas)
        phi_min = np.deg2rad(phi_min)
        phi_max = np.deg2rad(phi_max)
        phi_tot = phi_max - phi_min
        all_phi, all_theta = [], []
        if regular_id:
            n_large_ring = get_n_per_ring(r_t, dl, phi_tot)
            d_phi = phi_tot / n_large_ring
            self.n_phi = n_large_ring
            phis = d_phi * np.arange(int(not incl_min), n_large_ring + int(incl_max)) + phi_min
            all_phi = np.repeat(phis, n_ring)
            all_theta = np.tile(thetas, n_large_ring)
        else:
            nLs = get_n_per_ring(r_P, dl, phi_ring=phi_tot)
            d_phis = phi_tot / nLs
            for theta, nL, d_phi in zip(thetas, nLs, d_phis):
                cur_phis = d_phi * np.arange(int(not incl_min), nL + int(incl_max)) + phi_min
                all_phi.append(cur_phis)
                cur_thetas = theta * np.ones_like(cur_phis)
                all_theta.append(cur_thetas)
            self.theta_at_phi_min = 0
            self.theta_at_phi_max = 0
            if incl_min:
                self.theta_at_phi_min = all_theta[0].min()
            all_theta = np.hstack(all_theta)
            all_phi = np.hstack(all_phi)
        self.phi = all_phi
        self.theta = all_theta
        ys = r_ring * np.sin(all_theta)
        xs = (r_t - r_ring * np.cos(all_theta)) * np.sin(all_phi)
        zs = (r_t - r_ring * np.cos(all_theta)) * np.cos(all_phi)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, plane=plane)

    def extend(self, xs, ys, zs, pos='end'):
        """Extend the torus at the start or the end"""
        assert xs.size == ys.size == zs.size
        if pos == 'end':
            self.xs = np.r_[self.xs, xs]
            self.ys = np.r_[self.ys, ys]
            self.zs = np.r_[self.zs, zs]
        elif pos == 'start':
            self.xs = np.r_[xs, self.xs]
            self.ys = np.r_[ys, self.ys]
            self.zs = np.r_[zs, self.zs]
        else:
            raise ValueError('pos can only be end or start')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dl = 0.2
    fig = plt.figure(figsize=(8, 8))
    # 1D gallery ====================
    ax0 = fig.add_subplot(221)

    line = Line(2, 'x', dl, 'line').rotate(-30, 'z')
    ax0.plot(line.xs, line.ys, 'o')

    arc = Arc(1.6, '[90, 180)', 0, 'XOY', dl, 'arc')
    ax0.plot(arc.xs, arc.ys, 'o')

    circle = Circle(1.5, 0, 'XOY', dl, 'circle').shift(x=1)
    ax0.plot(circle.xs, circle.ys, 'o')

    symm_lines = SymmLines(2, 'z', 0.6, dl, 'symm_lines').shift(x=-2.5, y=-1.8)
    ax0.plot(symm_lines.xs, symm_lines.ys, 'o')

    torus2D = Torus2D(0.4, 1.2, dl, plane='XOY', phi_range='[0,270)').shift(x=-1, y=3)
    ax0.plot(torus2D.xs, torus2D.ys, 'o')
    ax0.axis('equal')
    # 2D gallery ====================
    ax1 = fig.add_subplot(223)

    filled_rectangle = FilledRectangle(1.7, 2, 0, 'z', dl).shift(x=-4)
    ax1.plot(filled_rectangle.xs, filled_rectangle.ys, 'o')
    thick_rectangle = ThickRectangle(1.7, 2, 2,
                                     0, 'z', dl).shift(x=-4, y=-3)
    ax1.plot(thick_rectangle.xs, thick_rectangle.ys, 'o')
    thick_ring = ThickRing(1.6, 1, dl, incl_inner=True, incl_outer=True,
                           axis='z').shift(y=1)
    ax1.plot(thick_ring.xs, thick_ring.ys, 'o')
    filled_circle = FilledCircle(1, dl, axis='z').shift(y=-2)
    ax1.plot(filled_circle.xs, filled_circle.ys, 'o')
    ax1.axis('equal')

    # 3D gallery ====================
    ax2 = fig.add_subplot(222, projection='3d')
    dl = 0.3
    block = Block(3, 4, 5, dl).shift(x=-5, y=-4)
    ax2.plot(block.xs, block.ys, block.zs, 'o', alpha=0.5, ms=2)
    #
    tube = CylinderSide(2, 10, dl, 'z').shift(x=-4, y=4)
    ax2.plot(tube.xs, tube.ys, tube.zs, 'o', alpha=0.5, ms=2)
    #
    torus = Torus(2, 5, dl, get_n_per_ring(2, dl), regular_id=True, plane='XOY', phi_range='[0,150)')
    ax2.plot(torus.xs, torus.ys, torus.zs, 'o', alpha=0.5, ms=2)

    thick_block_wall = ThickBlockWall(3, 4, 5, 2, dl).shift(z=3)
    ax2.plot(thick_block_wall.xs, thick_block_wall.ys, thick_block_wall.zs, 'o', alpha=0.5, ms=2)

    ax2.view_init(elev=46, azim=-42, roll=0)
    ax2.axis('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    # Transformation
    ax3 = fig.add_subplot(224, projection='3d')
    tube = tube.shift(x=4)
    ax3.plot(tube.xs, tube.ys, tube.zs, 'o', alpha=0.5, ms=2)
    for i in range(3):
        tube_rot = tube.rotate(90 * (i + 1), 'x')
        ax3.plot(tube_rot.xs, tube_rot.ys, tube_rot.zs, 'o', alpha=0.5, ms=2)
    ax3.plot([-5, 5], [0, 0], [0, 0], '--')

    x_plane = np.linspace(10, 20)
    y_plane = np.linspace(-10, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.full_like(X_plane, 0)
    ax3.plot_surface(X_plane, Y_plane, Z_plane, rstride=3, cstride=3, alpha=0.5)
    torus = Geometry().set_coord(*transform_coordinate(torus.xs, torus.ys, torus.zs, plane='XOZ')).shift(x=15, z=5)
    ax3.plot(torus.xs, torus.ys, torus.zs, 'o', alpha=0.5, ms=2)
    torus_mirror = torus.mirror('XOY', 0)
    ax3.plot(torus_mirror.xs, torus_mirror.ys, torus_mirror.zs, 'o', alpha=0.5, ms=2)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.view_init(elev=24, azim=-51, roll=0)
    ax3.axis('equal')

    plt.show()
