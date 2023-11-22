"""
Check how well the necromancer scales with the number of nodes and 
if there are any artifacts in the results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr
from necromancer import NumberNecromancer
import subprocess
import os

# Check if ./.tmp exists if not create it
if not os.path.exists("./.tmp"):
    os.mkdir("./.tmp")

# --- Plot 1: Scaling with rank ---
N_dispersion = []  # Slightly deviates due to floor division when chunking
result_dispersion = []
t_exec_dispersion = []
# Gather data
if True:
    node_range = np.arange(1, 152)
    # Result dispersion across nodes
    result_dispersion = []
    t_exec_dispersion = []
    for i, node in enumerate(node_range):
        print(f"Running {i + 1} of {len(node_range)}")
        # Spawn the process
        cmd = f"mpiexec -np {node} python -m mpi4py ./StarShapes/to_spawn.py"
        subprocess.call(cmd.split(" "))

    exit()
else:
    # Load data
    with np.load("./StarShapes/Results/node_scaling.npz") as data:
        node_range = data["node_range"]
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        result_dispersion = data["result_dispersion"]
# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
t_exec_disp = []
for i, n in enumerate(t_exec_dispersion):
    l = list("".join(n[1:-1]))
    t_exec_disp.append(l)

print(t_exec_disp)

print(t_exec_disp)
cm = pl.colorbrewer.sequential.PuRd_7.mpl_colormap
colors = cmr.take_cmap_colors(cm, 7, cmap_range=(0.15, 0.85), return_fmt="hex")

# Plot 1: Computation time vs. number of nodes
ax[0].scatter(node_range, t_exec_dispersion, color=colors[2], s=4)
ax[0].plot(node_range, t_exec_dispersion, color=colors[0], lw=0.75)


plt.tight_layout()
plt.show()
