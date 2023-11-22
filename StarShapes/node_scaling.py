"""
Check how well the necromancer scales with the number of nodes and 
if there are any artifacts in the results.
"""
import numpy as np
import matplotlib.pyplot as plt
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
if False:
    node_range = np.arange(1, 152)
    # Result dispersion across nodes
    result_dispersion = []
    t_exec_dispersion = []
    for i, node in enumerate(node_range):
        print(f"Running {i + 1} of {len(node_range)}")
        # Spawn the process
        subprocess.call(["mpiexec", "-n", str(node), "python", "to_spawn.py"])
        # Load the data
        with open("./.tmp/temp.txt", "r") as f:
            n_in = list(f.readline())
            n_tot = list(f.readline())
            t_exec = list(f.readline())
        
        # Append the data
        result_dispersion.append(n_in)
        t_exec_dispersion.append(t_exec)

        # This should be constant but just in case
        N_dispersion.append(n_tot)

    if not os.path.exists("./StarShapes/Results"):
                os.mkdir("./StarShapes/Results")
    np.savez("./StarShapes/Results/node_scaling.npz", node_range=node_range, N_dispersion=N_dispersion,
              t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion)
else:
    # Load data
    with np.load("./StarShapes/Results/node_scaling.npz") as data:
        node_range = data["node_range"]
        N_dispersion = data["N_dispersion"]
# Plot

