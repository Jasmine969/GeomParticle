import numpy as np
from lammps import lammps
import geoparticle as gp

dl = 1e-4
l_box = 0.01
h_box = 0.006
l_water = 0.003
h_water = 0.0045
wall = gp.ThickRectangle(l_box, h_box, 2, 0, 'z', dl, 'wall')
water = gp.FilledRectangle(l_water, h_water, 0, 'z', dl, name='water').shift(x=dl, y=dl)
gas = gp.FilledRectangle(
    l_box - 2 * dl, h_box - 2 * dl, 0, 'z', dl, name='gas'
).shift(x=dl, y=dl).subtract(water, rmax=1e-6)

fac_buf = 1
xlo = wall.xs.min() - wall.xs.max() * fac_buf
xhi = wall.xs.max() * (1 + fac_buf)
ylo = wall.ys.min() - wall.ys.max() * fac_buf
yhi = wall.ys.max() * (1 + fac_buf)
zlo = -3e-4
zhi = 3e-4
lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])
lmp.commands_string(f"""
dimension	2
atom_style  rheo
units		si
newton	 	on
boundary	f f p
comm_modify vel yes
region      simulation_box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}
create_box  3 simulation_box
""")
n_atoms_wall = wall.size
lmp.create_atoms(
    n_atoms_wall, np.arange(n_atoms_wall) + 1 + lmp.get_natoms(),
    np.full(n_atoms_wall, 1, dtype=int), wall.flatten_coords)
n_atoms_water = water.size
lmp.create_atoms(
    n_atoms_water, np.arange(n_atoms_water) + 1 + lmp.get_natoms(),
    np.full(n_atoms_water, 2, dtype=int), water.flatten_coords)
n_atoms_gas = gas.size
lmp.create_atoms(
    n_atoms_gas, np.arange(n_atoms_gas) + 1 + lmp.get_natoms(),
    np.full(n_atoms_gas, 3, dtype=int), gas.flatten_coords)
n_atoms_all = lmp.get_natoms()
log_n_atom = f'n_atoms_wall: {n_atoms_wall},' \
             f' n_atom_water: {n_atoms_water}' \
             f' n_atoms_all: {n_atoms_all}.'
print(log_n_atom)

m0_fluid = dl ** 2 * 993
m0_gas = dl ** 2 * 1.1
lmp.commands_string(f"""
mass 1*2 {m0_fluid}
mass 3 {m0_gas}
run 0
write_data gas_liquid_dam2D.data
""")

# check atom overlapping
lmp.commands_string(f"""
pair_style      zero {dl * 2}
pair_coeff      * *
neighbor        {dl * 0.1} bin
delete_atoms    overlap {dl * 0.8} all all
""")
n_atoms_all_now = lmp.get_natoms()
if n_atoms_all == n_atoms_all_now:
    print('Congrats, no atoms are overlapped!')
else:
    raise RuntimeError(f'{n_atoms_all - n_atoms_all_now} atoms are overlapped!')
