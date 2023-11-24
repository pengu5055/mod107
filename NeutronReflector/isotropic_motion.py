"""
Model of a neutron reflector with isotropic (planar) motion
of the neutron allowed.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# import palettable as pl
import cmasher as cmr
from necromancer import NumberNecromancer
import os

# Consts
D = 1
L = D/2
N = 1e6

def condition(pairs, D=D, L=L):
    results = []
    for pair in pairs:
        exp_pre, theta_pre = pair
        theta = np.arccos(2 * theta_pre - 1)
        exp = -1/L * np.log(1 - exp_pre)
        pos = D/2 + exp * np.cos(theta)

        if pos > D:
            results.append(1)
        elif pos < 0:
            results.append(-1)
        else:
            results.append(0)

    return np.array(results)


# Iterate over D values
if True:
    D_range = np.linspace(0.01, 10, 152)
    N_dispersion = []
    t_exec_dispersion = []
    counter_state_dispersion = []
    for i, D in enumerate(D_range):
        cond = lambda x: condition(x, D=D)
        nn = NumberNecromancer(cond, num_samples=N, num_dimensions=2, domain=[0, 1], quiet_slaves=True)
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(D_range)}")
        n_in, n_tot, t_exec, res = nn.run()
        N_dispersion.append(n_tot)
        counter_state_dispersion.append(n_in)
        t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        if not os.path.exists("./NeutronReflector/Results"):
            os.mkdir("./NeutronReflector/Results")
    
        np.savez("./NeutronReflector/Results/D_scale_isotropic_recalc.npz", N_dispersion=N_dispersion,
                    counter_state_dispersion=counter_state_dispersion, t_exec_dispersion=t_exec_dispersion)
    
    nn.burry()
    exit()
else:
    # Load data
    with np.load("./NeutronReflector/Results/D_scale_isotropic.npz") as data:
        N_dispersion = data["N_dispersion"]
        t_exec_dispersion = data["t_exec_dispersion"]
        counter_state_dispersion = data["counter_state_dispersion"]
    
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
nodes = np.arange(1, 152)
D_range = np.linspace(0.01, 10, 152)

T = counter_state_dispersion / N_dispersion

cm_0 = pl.scientific.sequential.Tokyo_20.mpl_colormap
norm = mpl.colors.Normalize(vmin=T.min(), vmax=T.max())
sm = plt.cm.ScalarMappable(cmap=cm_0, norm=norm)
cbar = fig.colorbar(sm, ax=ax[0])
cbar.set_label("Transmisivity")
ax[0].imshow(T, extent=[nodes[0], nodes[-1], D_range[0], D_range[-1]], aspect="auto", origin="lower", cmap=cm_0, norm=norm)
ax[0].set_xlabel("Node")
ax[0].set_xlabel("Thickness of reflector D")
ax[0].set_title("Transmisivity of neutron reflector")

cm_1 = pl.scientific.sequential.Acton_20.mpl_colormap
norm = mpl.colors.Normalize(vmin=t_exec_dispersion.min(), vmax=t_exec_dispersion.max())
sm = plt.cm.ScalarMappable(cmap=cm_1, norm=norm)
cbar = fig.colorbar(sm, ax=ax[1])
cbar.set_label("Execution time [s]")

ax[1].imshow(t_exec_dispersion, extent=[nodes[0], nodes[-1], D_range[0], D_range[-1]], aspect="auto", origin="lower", cmap=cm_1, norm=norm)
ax[1].set_xlabel("Node")
ax[1].set_ylabel("Thickness of reflector D")
ax[1].set_title("Execution time across nodes")

plt.tight_layout()
plt.savefig("./NeutronReflector/Images/isotropic_motion.png", dpi=700)
plt.show()