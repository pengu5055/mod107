"""
Model of a neutron reflector with isotropic (planar) motion
of the neutron allowed.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import palettable as pl
import cmasher as cmr
from necromancer import NumberNecromancer
import os

# Consts
D = 1
L = D/2
N = 1e6

def condition(pairs, D=D, L=L):
    results = []
    for pair in pairs:
        exp_pre, theta_pre = pair
        theta = np.arccos(2 * theta_pre - 1)
        exp = -1/L * np.log(1 - exp_pre)
        pos = 0.5 + exp * np.cos(theta)

        if pos < D:
            results.append(True)
        else:
            results.append(False)

    return np.array(results)


# Iterate over D values
if True:
    D_range = np.linspace(0.01, 10, 152)
    N_dispersion = []
    t_exec_dispersion = []
    counter_state_dispersion = []
    for i, D in enumerate(D_range):
        cond = lambda x: condition(x, D=D)
        nn = NumberNecromancer(cond, num_samples=N, num_dimensions=2, domain=[0, 1], quiet_slaves=True)
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(D_range)}")
        n_in, n_tot, t_exec, res = nn.run()
        N_dispersion.append(n_tot)
        counter_state_dispersion.append(n_in)
        t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
    
        np.savez("./NeutronReflector/Results/D_scale_isotropic.npz", N_dispersion=N_dispersion,
                    counter_state_dispersion=counter_state_dispersion, t_exec_dispersion=t_exec_dispersion)
    
    nn.burry()
    exit()
else:
    # Load data
    with np.load("./NeutronReflector/Results/D_scale_isotropic.npz") as data:
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        counter_state_dispersion = data["counter_state_dispersion"]
    
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))



plt.show()
