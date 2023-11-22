"""
Calculate the mass of the weird star shape that was 
given in the assignment. It is assumed here that the density
is constant throughout the shape.
"""
import numpy as np
from necromancer import NumberNecromancer
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr

# Let's start of with 3 dimensions and complicate it if time allows
# Maybe display the shape in 3D just for fun
def condition(pairs):
    results = []
    for pair in pairs:
        x, y, z = pair
        results.append(np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1)
    return np.array(results)

# EXAMPLE:
# Call buddy necromancer
# nn = NumberNecromancer(condition, num_samples=10000000, num_dimensions=3, domain=[-1, 1])
# Run the necromancer
# n_in, n_tot, t = nn.run()

# --- Plot 1: Scaling with N ---
if False:
    N_range = np.logspace(2, 10, 20)
    # Result dispersion across nodes
    N_dispersion = []  # Slightly deviates due to floor division when chunking
    t_exec_dispersion = []
    result_dispersion = []
    for i, N in enumerate(N_range):
        nn = NumberNecromancer(condition, num_samples=N, num_dimensions=3, domain=[-1, 1], quiet_slaves=True)
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(N_range)}")
        n_in, n_tot, t_exec, _ = nn.run()
        if nn.rank == 0:
            N_dispersion.append(n_tot)
            result_dispersion.append(n_in)
            t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./StarShapes/Results"):
            os.mkdir("./StarShapes/Results")
        np.savez("./StarShapes/Results/N_scaling.npz", N_range=N_range, N_dispersion=N_dispersion, 
                t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion)
    nn.burry()
    exit()

else:
    # Load data
    with np.load("./StarShapes/Results/N_scaling.npz") as data:
        N_range = data["N_range"]
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        result_dispersion = data["result_dispersion"]
    
# Plot
plt.rcParams["font.family"] = "IBM Plex Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "medium"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titleweight"] = "medium"

cm = pl.colorbrewer.sequential.PuRd_7.mpl_colormap
colors = cmr.take_cmap_colors(cm, 7, cmap_range=(0.15, 0.85), return_fmt="hex")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))