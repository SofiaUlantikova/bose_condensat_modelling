from src.base.scripts import calculate_dynamics
from src.parameters.default_params import oscillator

dt = 1e-8

ts_eval = [(3e-7, 1e-8)]

additional_params = [i+1 for i in range(8)]

additional_keys = ['random_coord', 'random_spin', 'random_phase']

def conv2bit(n):
    return [b == '1' for b in bin(n)[2:].rjust(4)]

for i in range(len(additional_params)):
    ps = oscillator.copy()
    current_params = conv2bit(additional_params[i])
    for j in range(len(additional_keys)):
        ps[additional_keys[i]] = current_params[i]
    for t_ev in ts_eval:
        calculate_dynamics(t_ev[0], dt, t_ev[1], 10, f'oscillation_{t_ev[1]}_{additional_params[i]}', ps)
    del ps