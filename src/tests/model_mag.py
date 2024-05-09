from src.base.scripts import calculate_dynamics
from src.parameters.default_params import b0s, mag_params, mag_fields

dt = 1e-8

ts_eval = [(3e-7, 1e-8), (3e-6, 1e-7)]

for i in range(len(mag_fields)):
    ps = mag_params.copy()
    ps['b0'] = mag_fields[i].update
    for t_ev in ts_eval:
        calculate_dynamics(t_ev[0], dt, t_ev[1], 10, f'mag_nonconst_{t_ev[1]}_{b0s[i]}', ps)
    del ps