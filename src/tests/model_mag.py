from src.base.scripts import calculate_dynamics
from src.parameters.default_params import b0s, mag_params, mag_fields

dt = 1e-7

ts_eval = [(3e-5, 1e-6)]
b0s = b0s
mag_fields = mag_fields

for i in range(len(mag_fields)):
    ps = mag_params.copy()
    ps['b0'] = mag_fields[i].update
    for t_ev in ts_eval:
        calculate_dynamics(t_ev[0], dt, t_ev[1], 10, f'mag_nonconst_b1_0.05_{t_ev[1]}_{b0s[i]}', ps)
    del ps