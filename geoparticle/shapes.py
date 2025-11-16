from warnings import warn
from .utils import *
from .ops import *

filter_guide = " Filter this warning by filterwarnings('ignore',message='.*quantized.*')"


class Line(Geometry):
    """1D line aligned to a principal axis, then reoriented."""

    def __init__(self, length: float, direction: str, dl: float, name=None):
        """
        Initialize a Line object.

        Args:
            length (float): Length of the line.
            direction (str): Principal axis direction ('x', 'y', or 'z').
            dl (float): Spacing between points along the line.
            name (str, optional): Name of the line. Defaults to None.
        """
        super().__init__(name=name or f'Line {self.get_counter()}', dimension=2)
        ys = _arange0_quantized(length, dl)
        self.length = float(ys[-1])
        if self.length != length:
            warn(f"{self.name}: length quantized from {length} to {self.length} (dl={dl})." + filter_guide)
        xs = np.zeros_like(ys)
        zs = np.zeros_like(ys)
        self.set_coord(*_transform_coordinate(xs, ys, zs, axis=direction))


class SymmLines(Geometry):
    """Two symmetric lines centered at origin and aligned along `direction`."""

    def __init__(self, length: float, direction: str, dist_half: float, dl: float, name=None):
        """
        Initialize a SymmLines object.

        Args:
            length (float): Length of each symmetric line.
            direction (str): Principal axis direction ('x', 'y', or 'z').
            dist_half (float): Half the distance between the two lines.
            dl (float): Spacing between points along the lines.
            name (str, optional): Name of the symmetric lines. Defaults to None.
        """
        super().__init__(name=name or f'SymmLines {self.get_counter()}', dimension=2)
        upper = Line(length, 'z', dl).shift(x=dist_half)
        lower = upper.mirror('YOZ', 0)
        me = Union((upper, lower))
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, axis=direction))


class Arc(Geometry):
    """2D circular arc on a plane."""

    def __init__(self, r: float, phi_range: str, plane_pos: float, plane: str, dl: float, name=None):
        """
        Initialize an Arc object.

        Args:
            r (float): Radius of the arc.
            phi_range (str): Angular range of the arc in degrees (e.g., '[0,90)').
            plane_pos (float): Position of the arc on the specified plane.
            plane (str): Plane in which the arc lies ('XOY', 'YOZ', or 'XOZ').
            dl (float): Spacing between points along the arc.
            name (str, optional): Name of the arc. Defaults to None.
        """
        super().__init__(name=name or f'Arc {self.get_counter()}', dimension=2)
        a_deg, b_deg, incl_min, incl_max, total_deg = _parse_interval_deg(phi_range)
        self.phi_tot_deg = total_deg
        phis = _discretize_arc_by_dl(r, dl, a_deg, b_deg, incl_min, incl_max)
        zs = r * np.cos(phis)
        xs = r * np.sin(phis)
        ys = np.full_like(xs, plane_pos)
        self.set_coord(*_transform_coordinate(xs, ys, zs, plane=plane))


class Torus2D(Geometry):
    """Two concentric arcs representing outer and inner boundary of a 2D torus segment."""

    def __init__(self, r_ring: float, r_t: float, dl: float, plane='XOZ', phi_range='[0,360)', name=None):
        """
        Initialize a Torus2D object.

        Args:
            r_ring (float): Radius of the torus tube.
            r_t (float): Radius of the torus centerline.
            dl (float): Spacing between points along the arcs.
            plane (str, optional): Plane in which the torus lies. Defaults to 'XOZ'.
            phi_range (str, optional): Angular range of the torus in degrees. Defaults to '[0,360)'.
            name (str, optional): Name of the torus. Defaults to None.
        """
        super().__init__(name=name or f'Torus2D {self.get_counter()}', dimension=2)
        inner = Arc(r_t - r_ring, phi_range, 0.0, 'XOZ', dl)
        outer = Arc(r_t + r_ring, phi_range, 0.0, 'XOZ', dl)
        me = Union((inner, outer))
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, plane=plane))


class Circle(Arc):
    """Full circle as an arc over [0,360)."""

    def __init__(self, r: float, plane_pos: float, plane: str, dl: float, name=None):
        """
        Initialize a Circle object.

        Args:
            r (float): Radius of the circle.
            plane_pos (float): Position of the circle on the specified plane.
            plane (str): Plane in which the circle lies ('XOY', 'YOZ', or 'XOZ').
            dl (float): Spacing between points along the circle.
            name (str, optional): Name of the circle. Defaults to None.
        """
        super().__init__(r, '[0,360)', plane_pos, plane, dl,
                         name=name or f'Circle {self.get_counter()}')


class Rectangle(Geometry):
    """2D rectangle boundary (inner dimensions)."""

    def __init__(self, length: float, width: float, plane_pos: float, axis: str, dl: float, name=None):
        """
        Initialize a Rectangle object.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            plane_pos (float): Position of the rectangle on the specified plane.
            axis (str): Axis orthogonal to the rectangle ('x', 'y', or 'z').
            dl (float): Spacing between points along the rectangle boundary.
            name (str, optional): Name of the rectangle. Defaults to None.
        """
        super().__init__(name=name or f'Rectangle {self.get_counter()}', dimension=2)
        z_bot = _arange0_quantized(length, dl)
        x_left = _arange0_quantized(width - dl, dl) + dl
        length_q, width_q = float(z_bot[-1]), float(x_left[-1])
        self.length, self.width = length_q, width_q
        if length_q != length or width_q != width:
            warn(f"{self.name}: size quantized to dl grid (dl={dl}):"
                 f" length {length} -> {length_q}, width {width} -> {width_q}" + filter_guide)
        # Build boundary without duplicating corners
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
        self.set_coord(*_transform_coordinate(xs, ys, zs, axis=axis))


class ThickRectangle(Geometry):
    """Rectangle wall with thickness inwards/outwards by dl-layers. Inner dims are (length,width)."""

    def __init__(self, length: float, width: float, n_thick: int,
                 plane_pos: float, axis: str, dl: float, name=None):
        """
        Initialize a ThickRectangle object.

        Args:
            length (float): Inner length of the rectangle.
            width (float): Inner width of the rectangle.
            n_thick (int): Number of thickness layers.
            plane_pos (float): Position of the rectangle on the specified plane.
            axis (str): Axis orthogonal to the rectangle ('x', 'y', or 'z').
            dl (float): Spacing between points along the rectangle boundary.
            name (str, optional): Name of the thick rectangle. Defaults to None.
        """
        super().__init__(name=name or f'ThickRectangle {self.get_counter()}', dimension=2)
        layers = []
        for i in range(n_thick):
            L = length + 2 * i * dl
            W = width + 2 * i * dl
            layer = Rectangle(L, W, plane_pos, 'y', dl).shift(x=-i * dl, z=-i * dl)
            layers.append(layer)
        me = Union(layers)
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, axis=axis))


class FilledRectangle(Geometry):
    """2D filled rectangle."""

    def __init__(self, length: float, width: float, plane_pos: float, axis: str, dl: float, name=None):
        """
        Initialize a FilledRectangle object.

        Args:
            length (float): Length of the rectangle.
            width (float): Width of the rectangle.
            plane_pos (float): Position of the rectangle on the specified plane.
            axis (str): Axis orthogonal to the rectangle ('x', 'y', or 'z').
            dl (float): Spacing between points within the rectangle.
            name (str, optional): Name of the filled rectangle. Defaults to None.
        """
        super().__init__(name=name or f'FilledRectangle {self.get_counter()}', dimension=2)
        z = _arange0_quantized(length, dl)
        x = _arange0_quantized(width, dl)
        Lq, Wq = float(z[-1]), float(x[-1])
        if Lq != length or Wq != width:
            warn(f"{self.name}: size quantized to dl grid (dl={dl}):"
                 f" length {length} -> {Lq}, width {width} -> {Wq}" + filter_guide)
        Z, X = np.meshgrid(z, x, indexing='xy')
        zs, xs = Z.ravel(), X.ravel()
        ys = np.full_like(xs, plane_pos, dtype=float)
        self.set_coord(*_transform_coordinate(xs, ys, zs, axis=axis))
        self.length, self.width = Lq, Wq


class ThickRing(Geometry):
    """2D ring region between inner and outer circles. Inner/outer rings can be included or excluded."""

    def __init__(self, r_out: float, r_in: float, dl: float,
                 incl_inner: bool = True, incl_outer: bool = True,
                 axis: str = 'y', adjust_dl: bool = False,
                 equal_size_per_circle: bool = False, name=None):
        """
        Initialize a ThickRing object.

        Args:
            r_out (float): Outer radius of the ring.
            r_in (float): Inner radius of the ring.
            dl (float): Spacing between points along the ring.
            incl_inner (bool): Whether to include the inner circle.
            incl_outer (bool): Whether to include the outer circle.
            axis (str, optional): Axis orthogonal to the ring. Defaults to 'y'.
            adjust_dl (bool, optional): Whether to adjust spacing for the inner circle. Defaults to False.
            equal_size_per_circle (bool, optional): Whether to use equal point count per circle. Defaults to False.
            name (str, optional): Name of the thick ring. Defaults to None.
        """
        super().__init__(name=name or f'ThickRing {self.get_counter()}', dimension=2)
        if r_out < r_in:
            raise ValueError('r_out must be >= r_in')
        self.dl = float(dl)
        n_inner = n_per_ring(r_in, self.dl) if r_in > 0 else 1
        if adjust_dl and r_in > 0:
            self.dl = float(_spacing_ring(r_in, n_inner))
        n_radial = int(round((r_out - r_in) / self.dl))
        self.r_out = r_in + n_radial * self.dl
        if abs(self.r_out - r_out) > 1e-6:
            warn(f"{self.name}: outer radius quantized from {r_out} to {self.r_out} (dl={self.dl})" + filter_guide)
        rs = np.arange(0, n_radial + 1) * self.dl + r_in
        if equal_size_per_circle and r_in > 0:
            n_per_rings = np.full_like(rs, n_inner, dtype=int)
        else:
            n_per_rings = n_per_ring(rs, self.dl)
        if not incl_inner and rs.size > 0:
            rs, n_per_rings = rs[1:], n_per_rings[1:]
        if not incl_outer and rs.size > 0:
            rs, n_per_rings = rs[:-1], n_per_rings[:-1]
        xs, zs = [], []
        if isinstance(n_per_rings, int):
            n_per_rings = np.array([n_per_rings])
        for r, n in zip(rs, n_per_rings):
            x, z = _ring_xy(int(n), float(r))  # x=cos, z=sin in default plane, will be remapped
            xs.append(x)
            zs.append(z)
        xs = np.asarray(np.hstack(xs)) if xs else np.array([], dtype=float)
        zs = np.asarray(np.hstack(zs)) if zs else np.array([], dtype=float)
        ys = np.zeros_like(xs)
        self.set_coord(*_transform_coordinate(xs, ys, zs, axis=axis))
        self.rs = rs
        self.n_per_rings = n_per_rings


class FilledCircle(ThickRing):
    """2D filled circle (ring with r_in=0 and inner/outer included)."""

    def __init__(self, r: float, dl: float, axis: str = 'y', name=None):
        """
        Initialize a FilledCircle object.

        Args:
            r (float): Radius of the circle.
            dl (float): Spacing between points within the circle.
            axis (str, optional): Axis orthogonal to the circle. Defaults to 'y'.
            name (str, optional): Name of the filled circle. Defaults to None.
        """
        super().__init__(r_out=r, r_in=0.0, dl=dl, incl_inner=True, incl_outer=True,
                         axis=axis, adjust_dl=False, equal_size_per_circle=False,
                         name=name or f'FilledCircle {self.get_counter()}')


class Block(Geometry):
    """3D block by stacking a filled rectangle along the z-axis."""

    def __init__(self, length: float, width: float, height: float, dl: float, name=None):
        """
        Initialize a Block object.

        Args:
            length (float): Length of the block along the x-axis.
            width (float): Width of the block along the y-axis.
            height (float): Height of the block along the z-axis.
            dl (float): Spacing between points in the grid.
            name (str, optional): Name of the block. Defaults to None.
        """
        super().__init__(name=name or f'Block {self.get_counter()}', dimension=3)
        layer = FilledRectangle(length, width, 0.0, 'z', dl)
        n_height = int(height / dl) + 1
        realized_height = (n_height - 1) * dl
        if realized_height != height:
            warn(f"{self.name}: height quantized from {height} to {realized_height} (dl={dl})" + filter_guide)
        self.height = realized_height
        me = Stack(layer, 'z', n_height, dl)
        self.set_coord(me.xs, me.ys, me.zs)


class ThickBlockWall(Geometry):
    """3D thick walls of a rectangular box, including side walls and top/bottom lids."""

    def __init__(self, length: float, width: float, height: float, n_thick: int, dl: float, name=None):
        """
        Initialize a ThickBlockWall object.

        Args:
            length (float): Inner length of the box.
            width (float): Inner width of the box.
            height (float): Height of the box.
            n_thick (int): Number of thickness layers for the walls.
            dl (float): Spacing between points in the grid.
            name (str, optional): Name of the thick block wall. Defaults to None.
        """
        super().__init__(name=name or f'ThickBlockWall {self.get_counter()}', dimension=3)
        side_layer = ThickRectangle(length, width, n_thick, 0.0, 'z', dl)
        n_height = int(height / dl) + 1
        side = Stack(side_layer, 'z', n_height + n_thick, dl).shift(z=-(n_thick - 1) * dl)
        lid_layer = FilledRectangle(length - 2 * dl, width - 2 * dl, 0.0, 'z', dl).shift(x=dl, y=dl)
        lid_lower = Stack(lid_layer, 'z', -n_thick, dl)
        z_mid = (n_height - 1) * dl / 2
        lid_upper = lid_lower.mirror('XOY', z_mid)
        me = Union((side, lid_lower, lid_upper))
        self.set_coord(me.xs, me.ys, me.zs)


class CylinderSide(Geometry):
    """Side surface of a cylinder."""

    def __init__(self, r: float, l_axis: float, dl: float, axis: str = 'y', name=None):
        """
        Initialize a CylinderSide object.

        Args:
            r (float): Radius of the cylinder.
            l_axis (float): Length of the cylinder along the specified axis.
            dl (float): Spacing between points in the grid.
            axis (str, optional): Axis along which the cylinder is oriented. Defaults to 'y'.
            name (str, optional): Name of the cylinder side. Defaults to None.
        """
        super().__init__(name=name or f'CylinderSide {self.get_counter()}', dimension=3)
        self.l_axis = float(l_axis)
        self.radius = float(r)
        self.n_axis = int(self.l_axis / dl) + 1
        y = np.arange(0, self.n_axis) * dl
        realized_l_axis = float(y[-1])
        if realized_l_axis != self.l_axis:
            warn(f"{self.name}: axis length quantized from {l_axis} to "
                 f"{realized_l_axis} (dl={dl})" + filter_guide)
        self.l_axis = realized_l_axis
        self.n_ring = int(n_per_ring(self.radius, dl))
        z_ring, x_ring = _ring_xy(self.n_ring, self.radius)
        zs = np.tile(z_ring, self.n_axis)
        xs = np.tile(x_ring, self.n_axis)
        ys = np.repeat(y, self.n_ring)
        self.set_coord(*_transform_coordinate(xs, ys, zs, axis=axis))

    @property
    def dl_in_ring(self) -> float:
        """
        Get the actual spacing between points on a ring.

        Returns:
            float: Spacing between points on the ring.
        """
        return float(_spacing_ring(self.r, self.n_ring))


class ThickCylinderSide(Geometry):
    """Thick wall of a cylinder."""

    def __init__(self, r_out, r_in, l_axis: float, dl: float,
                 axis: str = 'y', name=None):
        """
        Initialize a ThickCylinderWall object.

        Args:
            r_out (float): Outer radius of the cylinder.
            r_in (float): Inner radius of the cylinder.
            l_axis (float): Length of the cylinder along the specified axis.
            dl (float): Spacing between points in the grid.
            axis (str, optional): Axis along which the cylinder is oriented. Defaults to 'y'.
            name (str, optional): Name of the thick cylinder wall. Defaults to None.
        """
        super().__init__(name=name or f'ThickCylinderWall {self.get_counter()}', dimension=3)
        n_axis = int(l_axis / dl) + 1
        realized_l_axis = (n_axis - 1) * dl
        if realized_l_axis != l_axis:
            warn(f"{self.name}: axis length quantized from {l_axis} to "
                 f"{realized_l_axis} (dl={dl})" + filter_guide)
        layer = ThickRing(r_out, r_in, dl)
        me = Stack(layer, 'y', n_axis, dl)
        self.radius = np.sqrt(me.xs ** 2 + me.zs ** 2)
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, axis=axis))


class FilledCylinder(Geometry):
    """Filled cylinder volume."""

    def __init__(self, r: float, l_axis: float, dl: float, axis: str = 'y', name=None):
        """
        Initialize a FilledCylinder object.

        Args:
            r (float): Radius of the cylinder.
            l_axis (float): Length of the cylinder along the specified axis.
            dl (float): Spacing between points in the grid.
            axis (str, optional): Axis along which the cylinder is oriented. Defaults to 'y'.
            name (str, optional): Name of the filled cylinder. Defaults to None.
        """
        super().__init__(name=name or f'FilledCylinder {self.get_counter()}', dimension=3)
        radial_layer = FilledCircle(r, dl, 'y')
        n_axis = int(l_axis / dl) + 1
        realized_l_axis = (n_axis - 1) * dl
        if realized_l_axis != l_axis:
            warn(f"{self.name}: axis length quantized from {l_axis} to "
                 f"{realized_l_axis} (dl={dl})" + filter_guide)
        self.l_axis = realized_l_axis
        me = Stack(radial_layer, 'y', n_axis, dl)
        self.radius = np.sqrt(me.xs ** 2 + me.zs ** 2)
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, axis=axis))


class TorusSurface(Geometry):
    """3D torus surface (tube radius r_ring around circle radius r_t)."""

    def __init__(self, r_ring: float, r_t: float, dl: float, n_ring: int = None,
                 plane='XOZ', phi_range='[180,270)', regular_id=False, name=None):
        """
        Args:
            r_ring (float): Radius of the torus tube.
            r_t (float): Radius of the torus centerline.
            dl (float): Spacing between points in the grid.
            n_ring (int): Number of points along the tube's cross-section.
            plane (str, optional): Plane in which the torus lies. Defaults to 'XOZ'.
            phi_range (str, optional): Angular range of the torus in degrees. Defaults to '[180,270)'.
            regular_id (bool, optional): Whether to use regular indexing for points. Defaults to False.
            name (str, optional): Name of the torus. Defaults to None.
        """
        super().__init__(name=name or f'TorusSurface {self.get_counter()}', dimension=3)
        if not (r_ring < r_t):
            raise ValueError('r_ring must be smaller than r_t')
        if r_ring == 0:
            me = Arc(r_t, phi_range, 0, plane, dl)
            self.set_coord(me.xs, me.ys, me.zs)
            return
        if n_ring is None:
            n_ring = n_per_ring(r_ring, dl)
        thetas = np.linspace(0, 2 * np.pi, int(n_ring), endpoint=False)  # section angle
        a_deg, b_deg, incl_min, incl_max, total_deg = _parse_interval_deg(phi_range)
        a = np.deg2rad(a_deg)
        b = np.deg2rad(b_deg)
        phi_tot = b - a

        if regular_id:
            n_large = int(n_per_ring(r_t, dl, phi_tot))
            d_phi = phi_tot / max(n_large, 1)
            phis = d_phi * np.arange(int(not incl_min), max(n_large, 1) + int(incl_max)) + a
            all_phi = np.repeat(phis, int(n_ring))
            all_theta = np.tile(thetas, phis.size)
        else:
            r_P = r_t - r_ring * np.cos(thetas)  # path radius of section point
            nLs = np.maximum(n_per_ring(r_P, dl, phi_tot), 1)
            all_phi, all_theta = [], []
            for theta, nL in zip(thetas, nLs):
                d_phi = phi_tot / nL
                cur_phi = d_phi * np.arange(int(not incl_min), nL + int(incl_max)) + a
                all_phi.append(cur_phi)
                all_theta.append(np.full_like(cur_phi, theta))
            all_phi = np.hstack(all_phi) if all_phi else np.array([], dtype=float)
            all_theta = np.hstack(all_theta) if all_theta else np.array([], dtype=float)

        ys = r_ring * np.sin(all_theta)
        xs = (r_t - r_ring * np.cos(all_theta)) * np.sin(all_phi)
        zs = (r_t - r_ring * np.cos(all_theta)) * np.cos(all_phi)
        self.set_coord(*_transform_coordinate(xs, ys, zs, plane=plane))
        self.n_theta = int(n_ring)
        self.n_phi = None if not regular_id else int(n_per_ring(r_t, dl, phi_tot))
        self.phi_tot_deg = total_deg


class ThickTorusWall(Geometry):
    """3D thick wall of a torus."""

    def __init__(self, r_in: float, r_t: float, n_thick: int, dl: float,
                 plane='XOZ', phi_range='[180,270)', name=None):
        """
        Args:
            r_in (float): Inner radius of the torus tube.
            n_thick (int): Number of thickness layers.
            r_t (float): Radius of the torus centerline.
            dl (float): Spacing between points in the grid.
            plane (str, optional): Plane in which the torus lies. Defaults to 'XOZ'.
            phi_range (str, optional): Angular range of the torus in degrees. Defaults to '[180,270)'.
            name (str, optional): Name of the thick torus wall. Defaults to None.
        """
        super().__init__(name=name or f'ThickTorusWall {self.get_counter()}', dimension=3)
        layers = []
        radii = []
        for i in range(n_thick):
            r_ring = r_in + i * dl
            layer = TorusSurface(r_ring, r_t, dl, phi_range=phi_range)
            layers.append(layer)
            radii.append(np.full_like(layer.xs, r_ring))
        self.radius = np.hstack(radii)
        me = Union(layers)
        self.set_coord(*_transform_coordinate(me.xs, me.ys, me.zs, plane=plane))


class FilledTorus(ThickTorusWall):
    """3D filled torus volume."""

    def __init__(self, r_ring: float, r_t: float, dl: float,
                 plane='XOZ', phi_range='[180,270)', name=None):
        """
        Args:
            r_ring (float): Outer radius of the torus tube.
            r_t (float): Radius of the torus centerline.
            dl (float): Spacing between points in the grid.
            plane (str, optional): Plane in which the torus lies. Defaults to 'XOZ'.
            phi_range (str, optional): Angular range of the torus in degrees. Defaults to '[180,270)'.
            name (str, optional): Name of the filled torus. Defaults to None.
        """
        n_thick = int(r_ring / dl) + 1
        self.r_ring = (n_thick - 1) * dl
        if abs(self.r_ring - r_ring) > 1e-6:
            warn(f"{self.name}: ring radius quantized from {r_ring} to {self.r_ring} (dl={dl})" + filter_guide)
        super().__init__(0, r_t, n_thick, dl, plane=plane, phi_range=phi_range)


class SphereSurface(Geometry):
    """
    3D sphere surface discretized with approximately uniform point spacing.
    Attempts to keep each point's spacing to its north/east/south/west neighbors close to dl.
    """

    def __init__(self, r: float, dl: float, name=None):
        """
        Args:
            r (float): Sphere radius
            dl (float): Particle spacing
            name (str, optional): Geometry name
        """
        super().__init__(name=name or f'Sphere {self.get_counter()}', dimension=3)
        self.r = float(r)
        self.dl = float(dl)

        if r < 0:
            raise ValueError('radius must not be negative')
        if dl <= 0:
            raise ValueError('particle spacing must be positive')
        if r < dl:
            self.set_coord(np.array([0]), 0, 0)
            return

        # Compute number of latitude layers
        n_lat = max(1, int(np.pi * r / dl) + 1)
        lat_angles = np.linspace(0, np.pi, n_lat)

        xs, ys, zs = [], [], []

        for i, theta in enumerate(lat_angles):
            # Radius of the current latitude ring
            r_lat = r * np.sin(theta)
            z = r * np.cos(theta)  # z coordinate corresponding to this latitude

            # Number of points on the current latitude ring
            if r_lat < 1e-10:  # near the poles
                if i == 0 or i == n_lat - 1:  # north or south pole
                    xs.append(0.0)
                    ys.append(0.0)
                    zs.append(z if i == 0 else -r)  # north or south pole
                continue

            n_lon = n_per_ring(r_lat, dl)

            # Generate angles in the longitude direction
            lon_angles = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

            # Compute coordinates on the current latitude ring
            x_ring = r_lat * np.cos(lon_angles)
            y_ring = r_lat * np.sin(lon_angles)
            z_ring = np.full_like(lon_angles, z)

            xs.extend(x_ring)
            ys.extend(y_ring)
            zs.extend(z_ring)

        # Set coordinates
        self.set_coord(xs, ys, zs)

        # Check and adjust pole positions
        self._adjust_poles()

        # Validate spacing
        self._validate_spacing()

    @property
    def radius(self):
        return np.sqrt((self.matrix_coords ** 2).sum(axis=1))

    def _adjust_poles(self):
        """Adjust pole positions to ensure spacing to neighboring points is close to dl."""
        from scipy.spatial import KDTree

        if self.size == 0:
            return

        coords = self.matrix_coords
        tree = KDTree(coords)

        # Find potential pole candidates (points with z near ±r)
        north_pole_candidates = np.where(coords[:, 2] > self.r - 1e-6)[0]

        # Adjust north pole
        if len(north_pole_candidates) == 0:
            # Add a north pole point
            new_coords = np.vstack([coords, [0, 0, self.r]])
            self.set_coord(new_coords[:, 0], new_coords[:, 1], new_coords[:, 2])
        elif len(north_pole_candidates) > 1:
            # Keep the point closest to the north pole, remove others
            distances = np.linalg.norm(coords[north_pole_candidates] - [0, 0, self.r], axis=1)
            keep_idx = north_pole_candidates[np.argmin(distances)]
            mask = np.ones(self.size, dtype=bool)
            mask[north_pole_candidates] = False
            mask[keep_idx] = True
            self.set_coord(self.xs[mask], self.ys[mask], self.zs[mask])

        # Adjust south pole (similar handling)
        coords = self.matrix_coords  # re-fetch coordinates
        south_pole_candidates = np.where(coords[:, 2] < -self.r + 1e-6)[0]

        if len(south_pole_candidates) == 0:
            new_coords = np.vstack([coords, [0, 0, -self.r]])
            self.set_coord(new_coords[:, 0], new_coords[:, 1], new_coords[:, 2])
        elif len(south_pole_candidates) > 1:
            distances = np.linalg.norm(coords[south_pole_candidates] - [0, 0, -self.r], axis=1)
            keep_idx = south_pole_candidates[np.argmin(distances)]
            mask = np.ones(self.size, dtype=bool)
            mask[south_pole_candidates] = False
            mask[keep_idx] = True
            self.set_coord(self.xs[mask], self.ys[mask], self.zs[mask])

    def _validate_spacing(self):
        """Validate spacing of points on the sphere surface."""
        from scipy.spatial import KDTree

        if self.size < 2:
            return

        coords = self.matrix_coords
        tree = KDTree(coords)

        # Check distance to the nearest neighbor for each point
        distances, indices = tree.query(coords, k=2)
        nearest_distances = distances[:, 1]  # exclude self

        avg_distance = np.mean(nearest_distances)
        min_distance = np.min(nearest_distances)
        max_distance = np.max(nearest_distances)

        tolerance = 0.2 * self.dl  # 20% tolerance

        if abs(avg_distance - self.dl) > tolerance:
            warn(f'{self.name}: average spacing {avg_distance:.4f} '
                 f'differs significantly from target spacing {self.dl:.4f}')

        if min_distance < self.dl * 0.8:
            warn(f'{self.name}: found minimum spacing {min_distance:.4f} '
                 f'much smaller than target spacing {self.dl:.4f}')

        if max_distance > self.dl * 1.2:
            warn(f'{self.name}: found maximum spacing {max_distance:.4f} '
                 f'much larger than target spacing {self.dl:.4f}')


class ThickSphere(Geometry):
    """Thick spherical shell between inner and outer spheres."""

    def __init__(self, r_out: float, r_in: float, dl: float, name=None):
        """
        Args:
            r_out (float): Outer radius of the spherical shell.
            r_in (float): Inner radius of the spherical shell.
            dl (float): Spacing between points in the grid.
            name (str, optional): Name of the thick sphere. Defaults to None.
        """
        super().__init__(name=name or f'ThickSphere {self.get_counter()}', dimension=3)
        if r_out < r_in:
            raise ValueError('r_out must be >= r_in')
        layers = []
        for r in np.arange(r_in, r_out + dl * 0.1, dl):
            layer = SphereSurface(r, dl)
            layers.append(layer)
        me = Union(layers)
        self.set_coord(me.xs, me.ys, me.zs)
        self.rs = np.sqrt((self.matrix_coords ** 2).sum(axis=1))


class FilledSphere(ThickSphere):
    """Filled sphere volume."""

    def __init__(self, r: float, dl: float, name=None):
        """
        Args:
            r (float): Radius of the filled sphere.
            dl (float): Spacing between points in the grid.
            name (str, optional): Name of the filled sphere. Defaults to None.
        """
        super().__init__(
            r_out=r, r_in=0, dl=dl,
            name=name or f'FilledSphere {self.get_counter()}'
        )
