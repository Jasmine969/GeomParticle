# GeomParticle
This is a Python programme for particle-based geometry construction. I use it for Smoothed Particle Hydrodynamics (SPH) simulation, especially LAMMPS-SPH and LAMMPS-RHEO packages.

Particles of specified geometries are typically created by the `lattice` command in LAMMPS, which can lead to rough surfaces when the particle spacing is not small enough. However, too small spacing can result in too many particles and thus increase the computational cost.

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20240402203849.png)

The case is the same when one creates atoms based on an external STL file (an example STL file exported by COMSOL is shown below):

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20240402204014.png)

To resolve this problem, I developed this package for easy construction of geometries where smooth surfaces are required. Miscellaneous geometries made up of particles with a spacing of `dl`, including 1D geometries (lines and curves):

<img width="1274" height="687" alt="image" src="https://github.com/user-attachments/assets/d90b36a4-730e-4bf9-bce4-b0c0897b3cc8" />

2D geometries (rectangles and circles):

<img width="1223" height="671" alt="image" src="https://github.com/user-attachments/assets/6d289a22-3ad0-43bc-967b-ab617543c2d3" />

3D geometries (blocks, cylinders, tori, and spheres):

<img width="1020" height="1040" alt="image" src="https://github.com/user-attachments/assets/4d658233-dce4-4962-84d2-5a810cab58c4" />

all of which can be surface, thick shells, or filled bodies.

Diverse operations are also provided, including translation, mirror, rotation, stack, clipping, union, intersection, and subtraction.

<img width="1077" height="1019" alt="image" src="https://github.com/user-attachments/assets/f274ebc7-2031-4a30-9d5b-25ee14092e34" />

Some utility functions are also provided.

Enjoy your journey with GeomParticle.
