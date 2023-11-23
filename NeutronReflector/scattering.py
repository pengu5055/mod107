"""
Number of scattering events needed for neutron to escape a slab of thickness D.
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
        pos = 0.5
        scatter_counter = 0
        exp_pre = pair
        while True:
            scatter_counter += 1
            exp = -1/L * np.log(1 - exp_pre)
            dir = -1 if np.random.rand() < 0.5 else 1
            newpos = pos + dir * exp
            if newpos > D:
                results.append(scatter_counter)
                break
            elif newpos < 0:
                results.append(-scatter_counter)
                break
            else:
                pos = newpos
                exp_pre = np.random.uniform(0, 1)

    return np.array(results)


def iso_condition(pairs):
    results = []
    for pair in pairs:
        pos = 0.5
        scatter_counter = 0
        while True:
            scatter_counter += 1
            exp_pre, theta_pre = pair
            theta = np.arccos(2 * theta_pre - 1)
            exp = -1/L * np.log(1 - exp_pre)
            newpos = pos + exp * np.cos(theta)
            if newpos > D:
                results.append(scatter_counter)
                break
            elif newpos < 0:
                results.append(-scatter_counter)
                break
            else:
                pos = newpos

    return np.array(results)
if False:
    nn = NumberNecromancer(condition, num_samples=1e6, num_dimensions=1, domain=[0, 1])
    n_in, n_tot, t_exec, _ = nn.run()
    lattice = nn.results_tot

    # From here process only on rank 0
    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
        
        np.savez("./NeutronReflector/Results/linear_scattering.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec,
                 lattice=lattice)

    nn = NumberNecromancer(iso_condition, num_samples=1e6, num_dimensions=2, domain=[0, 1])
    n_in, n_tot, t_exec, _ = nn.run()
    lattice = nn.results_tot

    # From here process only on rank 0
    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
        
        np.savez("./NeutronReflector/Results/iso_scattering.npz", n_in=n_in, n_tot=n_tot, t_exec=t_exec,
                 lattice=lattice)
    
    nn.burry()
    exit()

else:
    # Load data
    with np.load("./NeutronReflector/Results/linear_scattering.npz") as data:
        n_in = data["n_in"]
        n_tot = data["n_tot"]
        t_exec = data["t_exec"]
        lattice = data["lattice"]
        print(lattice.shape)

    with np.load("./NeutronReflector/Results/iso_scattering.npz") as data:
        n_in_iso = data["n_in"]
        n_tot_iso = data["n_tot"]
        t_exec_iso = data["t_exec"]
        lattice_iso = data["lattice"]
        print(lattice_iso.shape)
    
# Plot
plt.rcParams["font.family"] = "IBM Plex Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "medium"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titleweight"] = "medium"

cm = pl.colorbrewer.sequential.PuRd_7.mpl_colormap
colors = cmr.take_cmap_colors(cm, 7, cmap_range=(0.2, 0.8), return_fmt="hex")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Flatten the lattice
# Truncate the lattice because it is too large (plotting takes too long)
lattice = np.concatenate(lattice)[::100]
T = np.where(lattice > 0, lattice, np.nan)
# Remove the negative values
T = T[~np.isnan(T)]
print(T.shape)


lattice_iso = np.concatenate(lattice_iso)[::100]
T_iso = np.where(lattice_iso >= 0, lattice_iso, np.nan)
# Remove the negative values
T_iso = T_iso[~np.isnan(T_iso)]
print(T_iso.shape)

ax[0].hist(T, bins=10, density=True, color=colors[6], edgecolor=colors[1], zorder=3)

ax[0].set_facecolor("#ebf9fc")
ax[0].grid(color="#737373", alpha=0.1, zorder=-1)
ax[0].set_xlabel("Number of Scattering Events")
ax[0].set_ylabel("Density")
ax[0].set_yscale("log")
ax[0].set_title("Linear Motion")

ax[1].hist(T_iso, bins=10, density=True, color=colors[6], edgecolor=colors[1], zorder=3)

ax[1].set_facecolor("#ebf9fc")
ax[1].grid(color="#737373", alpha=0.1, zorder=-1)
ax[1].set_xlabel("Number of Scattering Events")
ax[1].set_ylabel("Density")
ax[1].set_yscale("log")
ax[1].set_title("Isotropic Motion")

plt.tight_layout()
plt.savefig("./NeutronReflector/Images/scattering.png", dpi=700)
plt.show()
