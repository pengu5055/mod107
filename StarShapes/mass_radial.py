"""
Calculate the mass of the star shape that was given in 
the task this time with a radial density profile.
"""
import numpy as np
from necromancer import NumberNecromancer
import os

# Consts
V_ROT = 1
R_MAX = 1
P_DEF = 0
N = 1e7

def v_tangential(x, y, z):
    '''
    Get tangential component of rotational velocity
    '''
    # Assume solid body rotation
    r = np.sqrt(x**2 + y**2 + z**2)
    return V_ROT * r / R_MAX

def condition(pairs, P=P_DEF):
    results = []
    for pair in pairs:
        x, y, z = pair
        check = np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1
        if check:
            integrand = 1 * np.sqrt(x**2 + y**2 + z**2)**P
        else:
            integrand = 0
        results.append(integrand)
    return np.array(results)

# --- Plot 1: Scaling with P ---
if True:
    P_range = np.linspace(0, 5, 100)
    # Result dispersion across nodes
    N_dispersion = []  # Slightly deviates due to floor division when chunking
    t_exec_dispersion = []
    result_dispersion = []
    integral_dispersion = []
    for i, P in enumerate(P_range):
        cond = lambda x: condition(x, P=P)
        nn = NumberNecromancer(cond, num_samples=N, num_dimensions=3, domain=[-1, 1])
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(P_range)}")
        n_in, n_tot, t_exec, res = nn.run()
        print
        N_dispersion.append(n_tot)
        result_dispersion.append(n_in)
        integral_dispersion.append(res)
        t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./StarShapes/Results"):
            os.mkdir("./StarShapes/Results")
        np.savez("./StarShapes/Results/P_scaling_radial.npz", N_range=P_range, N_dispersion=N_dispersion, 
                t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion,
                integral_dispersion=integral_dispersion)
nn.burry()
