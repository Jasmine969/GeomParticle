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


def get_spacing_ring(r, n):
    """
    Calculate the spacing of neighboring particles of a ring
    by the radius and particle count.
    :param r: radius
    :param n: particle count
    :return: spacing
    """
    d = r * np.sqrt(2 * (1 - np.cos(2 * np.pi / n)))
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
    :param kwargs: axis or plane, as described in plane2axis
    :return: xs, ys, zs in the new coordinate system
    """
    plane2axis_up = {'XOY': 'x', 'YOZ': 'y', 'XOZ': 'z'}
    assert len(kwargs) == 1
    if 'axis' in kwargs:
        axis_up = kwargs['axis']
        assert axis_up in ['x', 'y', 'z']
    elif 'plane' in kwargs:
        plane = kwargs['plane']
        assert plane in plane2axis_up.keys()
        axis_up = plane2axis_up[plane]
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
        """获取当前类的计数器值"""
        return cls.counter

    @property
    def size(self):
        return self.xs.size

    @property
    def flatten_coords(self):
        return np.c_[self.xs, self.ys, self.zs].flatten()

    def set_coord(self, xs, ys, zs):
        """At most one integer among xs, ys, and zs"""
        from collections import Counter
        assert Counter([type(each) for each in [xs, ys, zs]])[int] < 2
        self.xs = xs
        self.ys = ys
        self.zs = zs
        if isinstance(xs, int):
            self.xs = np.full_like(ys, xs)
        elif isinstance(ys, int):
            self.ys = np.full_like(xs, ys)
        elif isinstance(zs, int):
            self.zs = np.full_like(xs, zs)
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


class Stack(Geometry):
    """Stack a 2D geometry along an axis"""

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


# 2D geometries =================================
class ThickRectangle(Geometry):
    """2D FilledRectangle with thickness.
    Sometimes the wall has to be thickened to avoid penetration."""

    def __init__(self, length, width, n_thick, plane_pos, axis, dl,
                 name=None):
        super(ThickRectangle, self).__init__(
            name=f'ThickRectangle {self.get_counter()}' if name is None else name
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
        if n_thick == 1:
            zs = np.r_[z_bot, z_left, z_top, z_right]
            xs = np.r_[x_bot, x_left, x_top, x_right]
            ys = np.full_like(xs, plane_pos)
        else:
            d = dict()
            xs, ys, zs = [], [], []
            for pos, axis, direction in zip(
                    ['bot', 'left', 'top', 'right'], ['x', 'z', 'x', 'z'], [-1, -1, 1, 1]):
                d[pos] = eval(f'Geometry().set_coord(x_{pos}, plane_pos, z_{pos})')
                d[pos] = Stack(d[pos], axis, direction * n_thick, dl)
                xs.append(d[pos].xs)
                ys.append(d[pos].ys)
                zs.append(d[pos].zs)
            xs = np.hstack(xs)
            ys = np.hstack(ys)
            zs = np.hstack(zs)
        self.xs, self.ys, self.zs = transform_coordinate(xs, ys, zs, axis=axis)
        print(self.log)


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
            adjust_dl=False, equal_size_per_circle=False,
            name=None
    ):
        """
        2D filled circle
        :param r: radius
        :param dl: spacing
        :param axis: axis of the circle. if axis is y, it is on the XOZ plane
        :param adjust_dl: whether to make the global dl = inner dl
        :param equal_size_per_circle: whether each ring has the same number of particles as the inner ring
        """
        super().__init__(
            r_out=r, r_in=0, dl=dl,
            incl_outer=True, incl_inner=True,
            axis=axis, adjust_dl=adjust_dl,
            equal_size_per_circle=equal_size_per_circle,
            name=f'FilledCircle {self.get_counter()}' if name is None else name
        )


# 3D Geometries ==================================
class Block(Geometry):
    def __init__(self, length, width, height, dl, name=None):
        """
        :param length: dz
        :param width: dx
        :param height: dy
        """
        super().__init__(name=f'Block {self.get_counter()}' if name is None else name)
        layer = FilledRectangle(length, width, 0, 'y', dl)
        n_height = int(height / dl)
        self.height = n_height * dl
        self.log = f'{self.name}: real/ori height {self.height}/{height}'
        me = Stack(layer, 'y', n_height, dl)
        self.set_coord(me.xs, me.ys, me.zs)
        print(self.log)


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

    def get_and_delete(self, ind: int, direction: str):
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
            ind_req = ind_all[:ind]
            ind_remain = ind_all[ind:]
        elif direction == 'larger':
            ind_req = ind_all[ind + 1:]
            ind_remain = ind_all[:ind + 1]
        else:
            raise ValueError('Direction can only be smaller or larger')
        x_req = coords[ind_req, :, 0].flatten()
        y_req = coords[ind_req, :, 1].flatten()
        z_req = coords[ind_req, :, 2].flatten()
        self.xs = coords[ind_remain, :, 0].flatten()
        self.ys = coords[ind_remain, :, 1].flatten()
        self.zs = coords[ind_remain, :, 2].flatten()
        self.n_axis = len(ind_remain)
        return x_req, y_req, z_req


class Torus(Geometry):
    def __init__(
            self, r_ring, r_t, dl, n_ring,
            plane='YOZ',
            phi_range='[180,270)',
            name=None
            # smallest_ID=None
    ):
        """
        3D Torus surface
        :param r_ring: radius of the smaller ring (section)
        :param r_t: radius of the larger circle
        :param dl: spacing
        :param n_ring: particle count per ring
        :param plane: XOY/YOZ/XOZ
        :param phi_range: radian range of the larger circle.
            Use interval notation: [/] is including and (/) is not
        """
        super().__init__(name=f'Torus {self.get_counter()}' if name is None else name)
        assert r_ring < r_t
        phi_min, phi_max = (float(each) for each in phi_range[1:-1].split(','))
        phi_tot = phi_max - phi_min
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
        nLs = get_n_per_ring(r_P, dl, phi_ring=phi_tot)
        d_phis = phi_tot / nLs
        all_phi, all_theta = [], []
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
        # if smallest_ID:
        #     from pandas import DataFrame
        #     df = DataFrame({'theta': all_theta, 'phi': all_phi})
        #     df = df.sort_values(by='phi').reset_index()[:n_ring]
        #     df['index'] += smallest_ID
        #     df.sort_values(by='theta', inplace=True)
        #     self.atom_id_boundary = df['index'].to_list()
        xs = r_ring * np.sin(all_theta)
        ys = (r_t - r_ring * np.cos(all_theta)) * np.sin(all_phi)
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
