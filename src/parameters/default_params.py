from src.base.phys_sys import Magnetic_variable
import numpy as np

b0s = [0, 1e-6, 1e-4, 1e-2, 0.1, 1]
freq_oscillation = 5e2*np.pi
mag_fields = [Magnetic_variable(b0, freq_oscillation) for b0 in b0s]

full_example_params = {'w' : freq_oscillation, 'rho' : 6e-7, 'nrho' : 60, 'ntheta' : 20,
                       'nphi' : 20, 'b1': 0.05, 'b2': -0.1, 'b0' : mag_fields[1].update,
                       'm': 86.9, 'N': 100, 'spin': 2, 'random_coord': False,
                       'random_spin': False, 'random_phase' : True, 'deg' : 3}

full_example_params['N'] = 0
full_example_params['density'] = 0.05 # 'N' and 'density' are two ways to set number of particles

basic_params = {'w' : freq_oscillation, 'rho' : 6e-7, 'nrho' : 60, 'ntheta' : 20, 'nphi' : 20, 'm': 86.9, 'deg' : 1}

oscillator = basic_params.copy()
oscillator['phi'] = 0
oscillator['theta'] = 0
oscillator['nrho'] = 100
oscillator['rho'] = 1e-6
oscillator['b1'] = 0
oscillator['b2'] = 0
oscillator['deg'] = 20

mag_params = basic_params.copy()
mag_params['N'] = 100
mag_params['b1'] = 0.05
mag_params['b2'] = -0.1
mag_params['deg'] = 3 # yeh, its cold)
mag_params['random_coord'] = False
mag_params['random_spin'] = False
mag_params['random_phase'] = False
mag_params['spin'] = 1