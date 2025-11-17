# geoparticle
*A Python package for geometry construction in particle-based simulation.*

I mainly use this package for geometry construction in LAMMPS, with some examples provided in the repository; of course, it can also be used for other software.

## Installation

### Installing from pypi

```bash
pip install geoparticle
```

### Installing from the source code

Download and enter the source code directory, then

```
pip install .
```

## Background

Particles of specified geometries are typically created by the `lattice` command in LAMMPS, which can lead to rough surfaces when the particle spacing is not small enough. However, too small spacing can result in too many particles and thus increase the computational cost.

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20240402203849.png)

The case is the same when one creates atoms based on an external STL file (an example STL file exported by COMSOL is shown below):

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117123120.png)

## Features

To resolve this problem, I developed this package for easy construction of geometries where smooth surfaces are required. Miscellaneous geometries are provided, including 1D geometries (lines and curves):

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117124159.png)

2D geometries (rectangles and circles):

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117124218.png)

3D geometries (blocks, cylinders, tori, and spheres):

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117124231.png)

all of which can be surface, thick shells, or filled bodies.

Diverse operations are also provided, including translation, mirror, rotation, stack, clipping, union, intersection, and subtraction.

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117124246.png)

Some utility functions are also provided.

## Quick start

`examples/gallery.py` provided detailed scripts to yield the geometries above.

Two more examples are given to shown how to couple geoparticle with LAMMPS. The first example is the 2D gas-liquid dam break, while the second is the 3D human duodenum whose particles are connected with bonds and angles (bonded particle method) to model the continuum.

![](https://fengimages-1310812903.cos.ap-shanghai.myqcloud.com/20251117163138.png)

## Limitations

The particle spacings may be not exactly as specified in order to create a smooth surface.

Resultant geometries of boolean operations can have more particles than expected in some cases, because

- For intersection and subtraction, only particles with distances smaller than `rmax` will be identified the same. Users should align particles of different geometries to get the expected results.
- For union, particles of all the given geometries will be collected to yield the union. Users should ensure no particles are overlapped.
