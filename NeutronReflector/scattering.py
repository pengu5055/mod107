"""
Number of scattering events needed for neutron to escape a slab of thickness D.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr
from necromancer import NumberNecromancer
import os

# Consts
D = 1
L = D/2

def condition(pairs, D=D, L=L):
    results = []
    for pair in pairs:
        pos = 0.5
        scatter_counter = 0
        exp_pre = pair
        while True:
            scatter_counter += 1
            exp = -1/L * np.log(1 - exp_pre)
            dir = -1 if np.random.rand() < 0.5 else 1
            newpos = pos + dir * exp
            if newpos > D:
                results.append(scatter_counter)
                break
            elif newpos < 0:
                results.append(-scatter_counter)
                break
            else:
                pos = newpos
                exp_pre = np.random.uniform(0, 1)

    return np.array(results)


def iso_condition(pairs):
    results = []
    for pair in pairs:
        pos = 0.5
        scatter_counter = 0
        while True:
            scatter_counter += 1
            exp_pre, theta_pre = pair
            theta = np.arccos(2 * theta_pre - 1)
            exp = -1/L * np.log(1 - exp_pre)
            newpos = pos + exp * np.cos(theta)
            if newpos > D:
                results.append(scatter_counter)
                break
            elif newpos < 0:
                results.append(-scatter_counter)
                break
            else:
                pos = newpos

    return np.array(results)
if True:
    nn = NumberNecromancer(condition, num_samples=1e6, num_dimensions=1, domain=[0, 1])
    n_in, n_tot, t_exec, _ = nn.run()
    lattice = nn.results_tot

    # From here process only on rank 0
    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
        
        np.savez("./NeutronReflector/Results/linear_scattering.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec,
                 lattice=lattice)

    nn = NumberNecromancer(iso_condition, num_samples=1e6, num_dimensions=2, domain=[0, 1])
    n_in, n_tot, t_exec, _ = nn.run()
    lattice = nn.results_tot

    # From here process only on rank 0
    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
        
        np.savez("./NeutronReflector/Results/iso_scattering.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec,
                 lattice=lattice)
    
    nn.burry()
    exit()

else:
    # Load data
    pass