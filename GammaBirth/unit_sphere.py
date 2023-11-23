"""
Gamma rays are born inside a sphere with a average free
path length of X% of the radius of the sphere. How many gamma rays
escape the sphere? 
"""
import numpy as np
from necromancer import NumberNecromancer
import os

# Consts
R = 1
X = 0.1

# Modify necromancer sample generation
#   - 

def condition(pairs, X=X):
    results = []
    for pair in pairs:
        theta_pre, r_pre = pair
        theta = np.arccos(2 * theta_pre - 1)
        r = r_pre**(1/3)
        free_path = np.random.exponential(scale=R * X)
        dist = - r * np.cos(theta) + np.sqrt(1 - (r/R)**2*(1-np.cos(theta)**2))
        if free_path > dist:  
            results.append(True)
        else:
            results.append(False)

    return np.array(results)

# Iterate over X values
if True:
    X_range = np.linspace(0, 1, 100)
    N_dispersion = []  
    t_exec_dispersion = []
    counter_state_dispersion = []
    for i, X in enumerate(X_range):
        cond = lambda x: condition(x, X=X)
        nn = NumberNecromancer(cond, num_samples=1e7, num_dimensions=2, domain=[0, 1], quiet_slaves=True)
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(X_range)}")
        n_in, n_tot, t_exec, res = nn.run()
        if nn.rank == 0:
            N_dispersion.append(n_tot)
            counter_state_dispersion.append(n_in)
            t_exec_dispersion.append(t_exec)
    
    if not os.path.exists("./GammaBirth/Results"):
        os.mkdir("./GammaBirth/Results")
    np.savez("./GammaBirth/Results/X_scaling.npz", X_range=X_range, N_dispersion=N_dispersion, 
            t_exec_dispersion=t_exec_dispersion, counter_state_dispersion=counter_state_dispersion)
    nn.burry()
    exit()
else:
    pass