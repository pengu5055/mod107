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

if False:
    nn = NumberNecromancer(condition, num_samples=1e8, num_dimensions=1, domain=[0, 1])
    n_in, n_tot, t_exec, _ = nn.run()
    
    # From here process only on rank 0
    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
        
        np.savez("./NeutronReflector/Results/linear_motion.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec)
    
    nn.burry()
    exit()

else:
    # Load data
    with np.load("./NeutronReflector/Results/linear_motion.npz") as data:
        n_in = data["n_in"]
        n_tot = data["n_tot"]
        t_exec = data["t_exec"]
    
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
nodes = np.arange(0, 152)

print(nodes.shape)
T = n_in/n_tot
ax[0].scatter(nodes, T, color=colors[5], s=4, label="T per node")
ax[0].plot(nodes, T, color=colors[3], lw=0.5, alpha=0.5)
ax[0].axhline(np.mean(T), color="k", ls="--", label="Mean T", alpha=0.8)
ax[0].axhline(np.mean(T) + np.std(T), color="k", ls=":", alpha=0.4)
ax[0].axhline(np.mean(T) - np.std(T), color="k", ls=":", alpha=0.4)
ax[0].fill_between(np.arange(0, 153), np.mean(T) + np.std(T), np.mean(T) - np.std(T),
                   label="1$\sigma$ band", color=colors[0], alpha=0.3)


ax[0].set_facecolor("#ebf9fc")
ax[0].grid(color="#737373", alpha=0.1)
ax[0].set_xlim(0, 153)
ax[0].set_xlabel("Node")
ax[0].set_ylabel("Transmisivity")
ax[0].legend(loc="upper left", frameon=False)
ax[0].set_title("Transmisivity per node w/ linear motion")


ax[1].scatter(nodes, t_exec, color=colors[5], s=4, label="Execution time per node")
ax[1].plot(nodes, t_exec, color=colors[3], lw=0.5, alpha=0.5)
ax[1].axhline(np.mean(t_exec), color="k", ls="--", label="Mean execution time", alpha=0.8)
ax[1].axhline(np.mean(t_exec) + np.std(t_exec), color="k", ls=":", alpha=0.4)
ax[1].axhline(np.mean(t_exec) - np.std(t_exec), color="k", ls=":", alpha=0.4)
ax[1].fill_between(np.arange(0, 153), np.mean(t_exec) + np.std(t_exec), np.mean(t_exec) - np.std(t_exec),
                   label="1$\sigma$ band", color=colors[0], alpha=0.3)

ax[1].set_facecolor("#ebf9fc")
ax[1].grid(color="#737373", alpha=0.1)
ax[1].set_xlim(0, 153)
ax[1].set_xlabel("Node")
ax[1].set_ylabel("Execution time [s]")
ax[1].legend(loc="lower right", frameon=False)
ax[1].set_title("Execution time @ 1e8 samples")


plt.tight_layout()
plt.savefig("./NeutronReflector/Results/linear_motion.png", dpi=700)
plt.show()