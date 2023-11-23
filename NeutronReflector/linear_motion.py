"""
Model of a neutron reflector with only linear motion of the neutron
allowed.
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
        exp_pre = pair
        exp = -1/L * np.log(1 - exp_pre)
        dir = -1 if np.random.rand() < 0.5 else 1
        pos = 0.5 + dir * exp

        if pos < D:
            results.append(True)
        else:
            results.append(False)

    return np.array(results)

nn = NumberNecromancer(condition, num_samples=1e8, num_dimensions=1, domain=[0, 1])
n_in, n_tot, t_exec, _ = nn.run()

# From here process only on rank 0
if nn.rank == 0:
    if not os.path.exists("./NeutronReflector/Results"):
        os.mkdir("./NeutronReflector/Results")
    
    np.savez("./NeutronReflector/Results/linear_motion.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec)