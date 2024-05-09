import numpy as np
from src.base.phys_sys import Simulation, Magnetic_variable
from src.base.plot_tools import Plot

def calculate_dynamics(t, dt, dt_eval, num_bigsteps, name_out, params=None, b0=lambda x: 0):
    if not params:
        # default params: extremely low temperature (deg = 1), binomial distribution of particles around center
        params = {'w': 500*np.pi, 'rho': 6e-7,
                    'nrho': 50, 'ntheta': 20, 'nphi': 20, 'b1': 0.05, 'b2': 0.05,
                    'b0': b0, 'm': 86.9, 'spin': 0, 'random_coord': False,
                    'random_spin': False, 'deg' : 1}
    sim = Simulation(params)
    for i in range(num_bigsteps):
        print(f'Start step {i}')
        sim.execute_propagation(t, dt, dt_eval)
    psis_dinam = np.concatenate(np.array(sim.solutions), axis=0)
    t_points = np.concatenate(np.array(sim.t_evals), axis=0)
    dtype = [('w', 'f'), ('rho', 'f'), ('nrho', 'i'), ('ntheta', 'i'), ('nphi', 'i'), ('b1', 'f'), ('b2', 'f'),
              ('m', 'f'), ('N', 'i'), ('spin', 'i'), ('random_coord', '?'), ('random_spin', '?'), ('deg', 'i')]
    params_list = []
    for d in dtype:
        try:
            params_list.append(params[d[0]])
        except KeyError:
            print(f'No such parameter: {d[0]}')
    np.savez(name_out+'.npy', paramiters=np.array([tuple(params_list)], dtype=dtype),
             time=t_points, psi_function=psis_dinam)

def explore_sim(filename, need_vort=True):
    with np.load(filename+'.npy.npz') as data:
        params = data['paramiters']
        t = data['time']
        psi = data['psi_function']
    params_dict = {'rho' : float(params['rho'][0]),
                   'nrho' : int(params['nrho'][0]), 'nphi' : int(params['nphi'][0]), 'ntheta' : int(params['ntheta'][0])}
    sim = Simulation(params_dict)
    t_end = t[-1]
    grid = sim.prop.grid
    w_plot = Plot(sim, np.absolute(psi)**2, 'Probability density', t_end, grid)
    real_plot = Plot(sim, np.real(psi), 'Re($\\psi$)', t_end, grid)
    if need_vort:
        vorticity = sim.eval_vorticity(psi)
        vort_plot = Plot(sim, vorticity, 'Vorticity', t_end, grid)
        return w_plot, real_plot, vort_plot
    return w_plot, real_plot
