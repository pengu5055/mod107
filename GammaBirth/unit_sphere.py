"""
Gamma rays are born inside a sphere with a average free
path length of X% of the radius of the sphere. How many gamma rays
escape the sphere? 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import palettable as pl
import cmasher as cmr
from necromancer import NumberNecromancer
import os

# Consts
R = 1
X = 0.1
N = 1e6


def condition(pairs, R=R, X=X):
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

# Iterate over X values AND R values
if False:
    X_range = np.linspace(0, 1, 100)
    R_range = np.linspace(0, 1, 100)
    mesh = []
    N_dispersion = []  
    t_exec_dispersion = []
    counter_state_dispersion = []
    for i, X in enumerate(X_range):
        mesh_col = []
        N_dispersion_col = []  
        t_exec_dispersion_col = []
        counter_state_dispersion_col = []
        for j, R in enumerate(R_range):
            cond = lambda x: condition(x, R=R, X=X)
            nn = NumberNecromancer(cond, num_samples=N, num_dimensions=2, domain=[0, 1], quiet_slaves=True)
            if nn.rank == 0:
                print(f"Running {i + 1}, {j + 1} of {len(X_range)}")
            n_in, n_tot, t_exec, res = nn.run()
            if nn.rank == 0:
                mesh_col.append((X, R))
                N_dispersion_col.append(n_tot)
                counter_state_dispersion_col.append(n_in)
                t_exec_dispersion_col.append(t_exec)
        if nn.rank == 0:
            mesh.append(mesh_col)
            N_dispersion.append(N_dispersion_col)
            counter_state_dispersion.append(counter_state_dispersion_col)
            t_exec_dispersion.append(t_exec_dispersion_col)
    
    if not os.path.exists("./GammaBirth/Results"):
        os.mkdir("./GammaBirth/Results")
    np.savez("./GammaBirth/Results/XR_scaling.npz", mesh=mesh, N_dispersion=N_dispersion, 
            t_exec_dispersion=t_exec_dispersion, counter_state_dispersion=counter_state_dispersion)
    nn.burry()
    exit()
else:
    # Load data
    with np.load("./GammaBirth/Results/XR_scaling.npz") as data:
        mesh = data["mesh"]
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        counter_state_dispersion = data["counter_state_dispersion"]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(counter_state_dispersion, origin="lower")

plt.show()
