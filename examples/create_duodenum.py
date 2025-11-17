import numpy as np
import geoparticle as gp
from lammps import lammps
from warnings import warn
from tqdm import trange
from functools import partial
import pandas as pd


def save_atomfile(ids, prop_per_atom, save_path='./atomfile'):
    n_atoms = prop_per_atom.size
    df = pd.DataFrame({'ID': ids, 'prop': prop_per_atom})
    df.to_csv(f'{save_path}.csv',
              index=False, sep='\t', header=[str(n_atoms), ''])


out = ''
lmp = lammps(cmdargs=['-screen', 'none', '-log', 'none'])

# parameters ====================
rho_fluid = 993
rho_wall = 1040
r_si = 0.03
dl = 3e-3
r_torus = 3 * r_si
n_ring_duo = 63
duo = gp.TorusSurface(r_ring=r_si, r_t=r_torus, dl=dl, n_ring=n_ring_duo, plane='XOZ',
                      phi_range='[90,330)', regular_id=True, name='duo')
phi = np.rad2deg(duo.phi)
theta = np.rad2deg(duo.theta)
ringID = np.repeat(np.arange(duo.n_phi), n_ring_duo)
save_atomfile(np.arange(duo.size) + 1, ringID, 'duodenum-ringID')
save_atomfile(np.arange(duo.size) + 1, theta, 'duodenum-theta')
duo.plot(c=phi)
# ==================== LAMMPS input
xlo = duo.xs.min() - r_si
xhi = duo.xs.max() + r_si
ylo = duo.ys.min() - r_si
yhi = duo.ys.max() + r_si
zlo = duo.zs.min() - r_si
zhi = duo.zs.max() + r_si
n_bond_type = 2
n_bond_per_atom = 4
lmp.commands_string(f"""
dimension	3
atom_style	hybrid bond angle rheo
units		si
newton	 	on
boundary	f f f
comm_modify vel yes
region      simulation_box block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi} 
create_box  1 simulation_box bond/types {n_bond_type} extra/bond/per/atom {n_bond_per_atom} &
            angle/types 1 extra/angle/per/atom 3
""")
n_atoms_duo = duo.size
lmp.create_atoms(
    n_atoms_duo, np.arange(n_atoms_duo) + 1 + lmp.get_natoms(),
    np.full(n_atoms_duo, 1, dtype=int), duo.flatten_coords)
log_n_atom = f'n_atoms_duo: {n_atoms_duo}'
lmp.commands_string(f"""
mass            1 1
variable        ringID atomfile duodenum-ringID.csv
variable        theta atomfile duodenum-theta.csv
variable        m0_wall atomfile duodenum-m0_wall.csv
set             group all rheo/rho {rho_fluid}
""")
# =============== create bonds
lmp.commands_string(f"""
pair_style      zero {dl * 2}
pair_coeff      * *
neighbor        {dl * 0.1} bin
special_bonds   lj/coul 0 1 1
""")
# connect each small ring
for ring_id in trange(duo.n_phi):
    lmp.commands_string(f"""
    variable        cur_ring atom v_ringID=={ring_id}
    group           cur_ring variable cur_ring
    create_bonds many cur_ring cur_ring 1 0.002 0.003
    variable        cur_ring delete
    group           cur_ring delete
    """)
n_bond_created, _ = lmp.gather_bonds()
print(n_bond_created)
# connect each large ring
thetas = np.linspace(0, 2 * np.pi, n_ring_duo, endpoint=False)
for theta_id in trange(n_ring_duo):
    cur_theta = thetas[theta_id]
    cur_dl = gp.spacing_ring(r_torus - r_si * np.cos(cur_theta),
                             duo.n_phi, phi_ring=duo.phi_tot)
    cur_dl *= 1.4
    lmp.commands_string(f"""
    variable        cur_ring atom v_theta=={np.rad2deg(cur_theta)}
    group           cur_ring variable cur_ring
    create_bonds many cur_ring cur_ring 2 0 {cur_dl}
    variable        cur_ring delete
    group           cur_ring delete
    """)
n_bond_expected = duo.size + (duo.size - n_ring_duo)
n_bond_created, _ = lmp.gather_bonds()

print('Connect per ring: angles')
get_wall_ID = partial(gp.get_wall_ID, n_per_ring=n_ring_duo, smallest_ID=1)
for i in trange(1, n_ring_duo + 1):
    for j in range(1, duo.n_phi + 1):
        lmp.command(f'create_bonds single/angle 1 '
                    f'{get_wall_ID(i, j)} {get_wall_ID(i + 1, j)} {get_wall_ID(i + 2, j)}')
n_angle_expected = n_ring_duo * duo.n_phi

filename = 'duo-wall'
lmp.commands_string(f"""
dump 1 all custom 1 {filename}.dump id type x y z v_ringID v_theta
run             0
write_data      {filename}.data
""")

if n_bond_created != n_bond_expected:
    warn(f'n_bond_created={n_bond_created},'
         f' but n_bond_expected={n_bond_expected}.')
else:
    print(f'{n_bond_expected} bonds are created successfully!')
