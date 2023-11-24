"""
Calculate the angular momentum of the star shape that was
given in the task this time with a radial density profile.
"""
import numpy as np
from necromancer import NumberNecromancer
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr
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
            integrand = np.sqrt(x**2 + y**2 + z**2) * np.sqrt(x**2 + y**2 + z**2)**P * v_tangential(x, y, z)
        else:
            integrand = 0
        results.append(integrand)
    return np.array(results)

# --- Plot 1: Scaling with P ---
if False:
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
        N_dispersion.append(n_tot)
        result_dispersion.append(n_in)
        integral_dispersion.append(res)
        t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./StarShapes/Results"):
            os.mkdir("./StarShapes/Results")
        np.savez("./StarShapes/Results/P_scaling_angular_radial.npz", N_range=P_range, N_dispersion=N_dispersion, 
                t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion,
                integral_dispersion=integral_dispersion)
    nn.burry()
    exit()
else:
    # Load data
    with np.load("./StarShapes/Results/P_scaling_angular_radial.npz") as data:
        P_range = data["N_range"]
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        result_dispersion = data["result_dispersion"]
        integral_dispersion = data["integral_dispersion"]


plt.rcParams["font.family"] = "IBM Plex Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.labelweight"] = "medium"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titleweight"] = "medium"

cm = pl.colorbrewer.sequential.PuRd_7.mpl_colormap
colors = cmr.take_cmap_colors(cm, 7, cmap_range=(0.15, 0.85), return_fmt="hex")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
nodes = np.arange(1, 152)

# Plot 1: Angular momentum Dispersion
cm_0 = pl.scientific.sequential.Tokyo_20.mpl_colormap
norm = mpl.colors.Normalize(vmin=integral_dispersion.min(), vmax=integral_dispersion.max())
sm = plt.cm.ScalarMappable(cmap=cm_0, norm=norm)
cbar = fig.colorbar(sm, ax=ax[0])
cbar.set_label("Angular momentum [arb. units]")

ax[0].imshow(integral_dispersion, extent=[nodes[0], nodes[-1], P_range[0], P_range[-1]], aspect="auto", origin="lower", cmap=cm_0, norm=norm)
ax[0].set_xlabel("Number of nodes")
ax[0].set_ylabel("Value of $P$")
ax[0].set_title("Distribution of angular\nmomentum across nodes")

# Plot 2: Execution time dispersion
cm_1 = pl.scientific.sequential.Acton_20.mpl_colormap
norm = mpl.colors.Normalize(vmin=t_exec_dispersion.min(), vmax=t_exec_dispersion.max())
sm = plt.cm.ScalarMappable(cmap=cm_1, norm=norm)
cbar = fig.colorbar(sm, ax=ax[1])
cbar.set_label("Execution time [s]")

ax[1].imshow(t_exec_dispersion, extent=[nodes[0], nodes[-1], P_range[0], P_range[-1]], aspect="auto", origin="lower", cmap=cm_1, norm=norm)
ax[1].set_xlabel("Node")
ax[1].set_ylabel("Value of $P$")
ax[1].set_title("Distribution of execution time across nodes")


plt.tight_layout()
plt.savefig("./StarShapes/Images/angular_radial_heatmaps.png", dpi=700)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: Different P values for angular momentum
values = np.average(integral_dispersion, axis=1)
t_avg = np.average(t_exec_dispersion, axis=1)
ax_add = ax[0].twinx()

ax[0].plot(P_range, values, color=colors[4], label="Angular momentum of star shape")
ax_add.plot(P_range, t_avg, color=colors[0], lw=0.7, alpha=0.3)
ax_add.axhline(np.mean(t_avg), color="k", ls="--", label="Mean execution time", alpha=0.4)
ax_add.scatter(P_range, t_avg, color=colors[2], label="Average execution time", s=2)


ax[0].set_facecolor("#ebf9fc")
ax[0].grid(color="#737373", alpha=0.1)
ax[0].set_xlabel("Value of $P$")
ax_add.set_ylabel("Execution time [s]")
ax[0].set_ylabel("Angular momentum [arb. units]")
ax[0].legend(loc="lower left", frameon=False)
ax_add.legend(loc=(0.3, 0.5), frameon=False, ncols=1, fontsize=10)
ax[0].set_title("Angular momentum of star shape\n for different values of $P$")

# Plot 2: Success rate per node
cm_2 = pl.scientific.sequential.Tokyo_20.mpl_colormap
sr = result_dispersion / N_dispersion

norm = mpl.colors.Normalize(vmin=sr.min(), vmax=sr.max())
sm = plt.cm.ScalarMappable(cmap=cm_2, norm=norm)
cbar = fig.colorbar(sm, ax=ax[1])
cbar.set_label("Number of accepted samples / Generated samples")

ax[1].imshow(sr, extent=[nodes[0], nodes[-1], P_range[0], P_range[-1]], aspect="auto", origin="lower", cmap=cm_2, norm=norm)
ax[1].set_xlabel("Node")
ax[1].set_ylabel("Value of $P$")
ax[1].set_title("Success rate across nodes")


plt.tight_layout()
plt.savefig("./StarShapes/Images/angular_radial.png", dpi=700)
plt.show()
