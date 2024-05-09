import sys
sys.path.append("/home/Sosuchka/PycharmProjects/bose_condensat_modelling/")

from src.base.scripts import explore_sim

probability, real_part, vorticity = explore_sim('./src/tests/mag_nonconst_1e-08_0')

grid = probability.grid
spins = [0, 1, 2]
thetas = [i for i in range(0, grid[1], grid[1]//4)]
phis = [i for i in range(0, grid[2], grid[2]//4)]
rhos = [i for i in range(0, grid[0], grid[0]//5)]

for theta in thetas:
    for phi in phis:
        probability.plot_spins_along_axis('rho', theta, phi)
        for rho in rhos:
            real_part.plot_spin_t_dep(rho, theta, phi)
            probability.plot_spin_t_dep(rho, theta, phi)
            vorticity.plot_spin_t_dep(rho, theta, phi)

for spin_projection in spins:
    probability.plot_3d(spin=spin_projection)
