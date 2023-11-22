"""
Calculate the angular momentum of the star shape that was
given in the task.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr
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
if False:
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
    exit()

# Load results
else:
    with np.load("./StarShapes/Results/N_scaling_angular.npz") as data:
        N_range = data["N_range"]
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        result_dispersion = data["result_dispersion"]
        integral_dispersion = data["integral_dispersion"]

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

# Sum the dispersion across nodes
N_ = np.sum(N_dispersion, axis=1)
result = np.sum(result_dispersion, axis=1)
integral = np.sum(integral_dispersion, axis=1)
t_exec = np.sum(t_exec_dispersion, axis=1)

# Plot 0
ax[0].plot(N_range, N_range, label="Ordered Samples", ls="--", color="#eb096f")
ax[0].plot(N_range, N_, label="Total Samples", color=colors[1], alpha=0.7)
ax[0].plot(N_range, result, label="Accepted Samples", color=colors[4])
# ax[0].plot(N_range, integral, label="Integral")


ax_add = ax[0].twinx()
ax_add.plot(N_range, t_exec, label="Execution Time", color=colors[6])
ax_add.set_ylabel("Execution Time [s]")
ax_add.legend(loc="lower right", frameon=False)
ax_add.set_xscale("log")
ax_add.set_yscale("log")
ax_add.set_ylim(0.01*t_exec.min(), 10*N_.max())
ax[0].set_facecolor("#ebf9fc")
ax[0].grid(color="#737373", alpha=0.1)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Number of Samples")
ax[0].legend(frameon=False)
ax[0].set_ylim(0.01*t_exec.min(), 10*N_.max())
ax[0].set_ylabel("Number of Samples")
ax[0].set_title("Exponential Scaling of Samples and Execution Time")


cm2 = pl.cartocolors.sequential.Magenta_7.mpl_colormap
colors2 = cmr.take_cmap_colors(cm2, len(result_dispersion.T), cmap_range=(0.15, 0.85), return_fmt="hex")
nodes = np.arange(0, len(result_dispersion.T))
# Plot 1 - Success rate per node
sr = np.sum(result_dispersion / N_dispersion, axis=0)
# for i, column in enumerate(result_dispersion.T):
#     ax[1].plot(N_range, column, color=colors2[i])
ax[1].plot(nodes, sr, color=colors2[0])

ax[0].set_facecolor("#ebf9fc")
ax[0].grid(color="#737373", alpha=0.1)

plt.show()

