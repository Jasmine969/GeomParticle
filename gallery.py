import matplotlib.pyplot as plt
import numpy as np
import my_geometry as gm

dl = 0.2
# 2D gallery ====================
fig = plt.figure(figsize=(16, 5))
ax1 = fig.add_subplot(131)

filled_rectangle = gm.FilledRectangle(1.7, 2, 0, 'z', dl).shift(x=-4)
ax1.plot(filled_rectangle.xs, filled_rectangle.ys, 'o')
thick_rectangle = gm.ThickRectangle(1.7, 2, 2,
                                 0, 'z', dl).shift(x=-4, y=-3)
ax1.plot(thick_rectangle.xs, thick_rectangle.ys, 'o')
thick_ring = gm.ThickRing(1.6, 1, dl, incl_inner=True, incl_outer=True,
                       axis='z').shift(y=1)
ax1.plot(thick_ring.xs, thick_ring.ys, 'o')
filled_circle = gm.FilledCircle(1, dl, axis='z').shift(y=-2)
ax1.plot(filled_circle.xs, filled_circle.ys, 'o')
ax1.axis('equal')

# 3D gallery ====================
ax2 = fig.add_subplot(132, projection='3d')
dl = 0.3
block = gm.Block(3, 4, 5, dl).shift(x=-5, y=-4)
ax2.plot(block.xs, block.ys, block.zs, 'o', alpha=0.5, ms=2)

tube = gm.CylinderSide(2, 10, dl, 'z').shift(x=-4, y=4)
ax2.plot(tube.xs, tube.ys, tube.zs, 'o', alpha=0.5, ms=2)

torus = gm.Torus(2, 5, dl, gm.get_n_per_ring(2, dl), plane='XOY', phi_range='[0,150)').shift(z=3)
ax2.plot(torus.xs, torus.ys, torus.zs, 'o', alpha=0.5, ms=2)

ax2.view_init(elev=46, azim=-42, roll=0)
ax2.axis('equal')

# Transformation
ax3 = fig.add_subplot(133, projection='3d')
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
torus = torus.rotate(90, 'y', axis_point1=(15, 0, 0)).shift(z=-4)
ax3.plot(torus.xs, torus.ys, torus.zs, 'o', alpha=0.5, ms=2)
torus_mirror = torus.mirror('XOY', 0)
ax3.plot(torus_mirror.xs, torus_mirror.ys, torus_mirror.zs, 'o', alpha=0.5, ms=2)

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.view_init(elev=24, azim=-51, roll=0)
ax3.axis('equal')

plt.show()