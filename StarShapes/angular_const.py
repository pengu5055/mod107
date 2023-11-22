"""
Calculate the angular momentum of the star shape that was
given in the task.
"""
import numpy as np
from necromancer import NumberNecromancer
import os

# Consts
V_ROT = 1
R_MAX = 1

def v_tangential(x, y, z):
    '''
    Get tangential component of rotational velocity
    '''
    # Assume solid body rotation
    r = np.sqrt(x**2 + y**2 + z**2)
    return V_ROT * r / R_MAX

def condition(pairs):
    results = []
    for pair in pairs:
        x, y, z = pair
        check = np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1
        if check:
            np.sqrt(x**2 + y**2 + z**2) * v_tangential(x, y, z)
        results.append(np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1)
    return np.array(results)

# --- Plot 1: Scaling with N ---
if True:
    N_range = np.logspace(2, 8, 20)
    # Result dispersion across nodes
    N_dispersion = []  # Slightly deviates due to floor division when chunking
    t_exec_dispersion = []
    result_dispersion = []
    integral_dispersion = []
    for i, N in enumerate(N_range):
        nn = NumberNecromancer(condition, num_samples=N, num_dimensions=3, domain=[-1, 1], quiet_slaves=True)
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(N_range)}")
        n_in, n_tot, t_exec, res = nn.run()
        if nn.rank == 0:
            N_dispersion.append(n_tot)
            result_dispersion.append(n_in)
            integral_dispersion.append(res)
            t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./StarShapes/Results"):
            os.mkdir("./StarShapes/Results")
        np.savez("./StarShapes/Results/N_scaling_angular_cluster.npz", N_range=N_range, N_dispersion=N_dispersion, 
                t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion,
                integral_dispersion=integral_dispersion)
nn.burry()
