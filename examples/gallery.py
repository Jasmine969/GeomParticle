import numpy as np
import geoparticle as gp
import matplotlib.pyplot as plt
from warnings import filterwarnings

filterwarnings('ignore',message='.*quantized.*')

dl = 0.2
# 1D gallery ====================
fig0 = plt.figure(figsize=(12, 6))
# line
ax01 = fig0.add_subplot(121)
line = gp.Line(2, 'x', dl, 'line').rotate(-30, 'z')
line.plot(ax01)
symm_lines = gp.SymmLines(2, 'z', 0.3, dl, 'symm_lines').shift(y=-2)
symm_lines.plot(ax01)
ax01.set_title('lines')
ax01.axis('equal')
# curves
ax02 = fig0.add_subplot(122)
arc = gp.Arc(1.6, '[90, 180)', 0, 'XOY', dl, 'arc')
arc.plot(ax02)

circle = gp.Circle(1.5, 0, 'XOY', dl, 'circle').shift(x=2)
circle.plot(ax02)

torus2D = gp.Torus2D(0.4, 1.2, dl, plane='XOY', phi_range='[0,270)').shift(x=-1, y=4)
torus2D.plot(ax02)

ax02.set_title('curves')
ax02.axis('equal')
# 2D gallery ====================
fig1 = plt.figure(figsize=(12, 6))
ax11 = fig1.add_subplot(121)
# rectangles
rectangle = gp.Rectangle(4, 1.2, 0, 'z', dl).shift(y=3)
rectangle.plot(ax11)
filled_rectangle = gp.FilledRectangle(1.7, 2, 0, 'z', dl)
filled_rectangle.plot(ax11)
thick_rectangle = gp.ThickRectangle(1.7, 2, 2, 0, 'z', dl).shift(x=2.5)
thick_rectangle.plot(ax11)
ax11.set_title('rectangles')
ax11.axis('equal')
# circles
ax12 = fig1.add_subplot(122)
circle = gp.Circle(1.5, 0, 'XOY', dl, 'circle').shift(x=3,y=-1)
circle.plot(ax12)
thick_ring = gp.ThickRing(1.6, 1, dl, incl_inner=True, incl_outer=True, axis='z').shift(y=1)
thick_ring.plot(ax12)
filled_circle = gp.FilledCircle(1, dl, axis='z').shift(y=-2)
filled_circle.plot(ax12)
ax12.set_title('circles')
ax12.axis('equal')

# 3D gallery ====================
fig2 = plt.figure(figsize=(10, 10))
dl = 0.4
# block
ax21 = fig2.add_subplot(221, projection='3d')
block = gp.Block(3, 4, 5, dl)
thick_block_wall = gp.ThickBlockWall(3, 4, 5, 2, dl).shift(x=5)
clipped_wall = gp.Clip(thick_block_wall, keep='negative', plane_point=(6.5, 2, 2.5), plane_normal=(0, 0, 1))
ax21 = block.plot(ax21)
ax21 = clipped_wall.plot(ax21, c=clipped_wall.zs)
ax21.set_title('Filled block and wall')
ax21.view_init(elev=33, azim=-76, roll=3)
# tube
ax22 = fig2.add_subplot(222, projection='3d')
tube = gp.CylinderSide(2, 10, dl, 'z')
thick_tube = gp.ThickCylinderSide(2, 1.5, 10, dl, 'z').shift(x=5)
water_column = gp.FilledCylinder(1.8, 10, dl, 'z').shift(x=10)
tube.plot(ax22)
thick_tube.plot(ax22, c=thick_tube.radius)
water_column.plot(ax22, c=water_column.radius)
ax22.set_title('cylinder side and filled cylinder')
# torus
ax23 = fig2.add_subplot(223, projection='3d')
torus_surface = gp.TorusSurface(2, 5, dl, plane='XOY', phi_range='[0,150)')
thick_torus = gp.ThickTorusWall(2, 5, 2, dl, plane='XOY', phi_range='[0,150)').shift(z=8)
filled_torus = gp.FilledTorus(2, 5, dl, plane='XOY', phi_range='[0,150)').shift(z=16)
torus_surface.plot(ax23, alpha=0.1)
thick_torus.plot(ax23, c=thick_torus.radius)
filled_torus.plot(ax23, c=filled_torus.radius)
ax23.set_title('torus')
# sphere
ax24 = fig2.add_subplot(224, projection='3d')
sphere_surface = gp.Clip(gp.SphereSurface(3, dl, 'sphere_surface'), keep='negative',
                      plane_point=(0,0,0), plane_normal=(0,0,1))
sphere_shell = gp.Clip(gp.ThickSphere(3, 2, dl), keep='negative',
                    plane_point=(0,0,0), plane_normal=(0,0,1))
sphere = gp.Clip(gp.FilledSphere(3, dl, 'sphere'), keep='negative',
              plane_point=(0, 0, 0), plane_normal=(0, 0, 1))
def calc_rs(coords):
    return np.sqrt((coords ** 2).sum(axis=1))

sphere_surface.plot(ax24)
sphere_shell.shift(x=8).plot(ax24, c=calc_rs(sphere_shell.matrix_coords))
sphere.shift(x=16).plot(ax24, c=calc_rs(sphere.matrix_coords))
ax24.view_init(elev=42, azim=-103, roll=-19)
ax24.set_title('Hollow and filled spheres')
for ax in (ax21, ax22, ax23, ax24):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')

# Transformation
fig3 = plt.figure(figsize=(10, 10))
# rotate
ax31 = fig3.add_subplot(221, projection='3d')
ax31.plot(tube.xs, tube.ys, tube.zs, 'o', alpha=0.5, ms=2)
for i in range(3):
    tube_rot = tube.rotate(90 * (i + 1), 'x', (0,2,-2))
    tube_rot.plot(ax31)
ax31.plot([-5, 5], [2, 2], [-2, -2], '--')
ax31.view_init(elev=12, azim=-16, roll=3)
ax31.set_title('rotate')
# mirror
ax32 = fig3.add_subplot(222, projection='3d')
x_plane = np.linspace(10, 20)
y_plane = np.linspace(-3, 10)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = np.full_like(X_plane, 0)
ax32.plot_surface(X_plane, Y_plane, Z_plane, rstride=3, cstride=3, alpha=0.5)
torus_surface = torus_surface.rotate(30, 'y').shift(x=15, z=5)
torus_surface.plot(ax32)
torus_mirror = torus_surface.mirror('XOY', 0)
torus_mirror.plot(ax32)
ax32.view_init(elev=11, azim=102, roll=-7)
ax32.set_title('mirror')
# intersect
ax33 = fig3.add_subplot(223, projection='3d')
torus_surface = gp.TorusSurface(2, 5, dl, gp.n_per_ring(2, dl), regular_id=False,
                     plane='XOZ', phi_range='[0,360)').shift(y=-5)
block =  gp.Block(3, 7, 15, dl).shift(x=-5,y=-8,z=-7)
intersect = gp.Intersect((torus_surface, block), rmax=dl).shift(x=-5,z=-15)
subtract = gp.Subtract(torus_surface, block, rmax=dl).shift(x=5, z=-15)
union = gp.Union((torus_surface, block)).shift(x=23, z=-15)
torus_surface.plot(ax33)
block.plot(ax33)
intersect.plot(ax33)
subtract.plot(ax33)
union.plot(ax33)
ax33.view_init(elev=15, azim=-106, roll=-3)
ax33.set_title('intersect, subtract, union')
# clip
ax34 = fig3.add_subplot(224, projection='3d')
thick_block_wall.plot(ax34)
clip = gp.Clip(thick_block_wall, keep='negative', plane_point=(1, 0, 3), plane_normal=(0, 0, 1)).shift(x=8)
x_plane = np.linspace(3, 11)
y_plane = np.linspace(-3, 6)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = np.full_like(X_plane, 3)
ax34.plot_surface(X_plane, Y_plane, Z_plane, rstride=3, cstride=3, alpha=0.5)
clip.plot(ax34, alpha=0.2)
ax34.view_init(elev=16, azim=-85, roll=-2)
ax34.set_title('clip')

for ax in (ax31, ax32, ax33, ax34):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')
plt.show()
