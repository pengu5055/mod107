"""
Combined plot for mass from radial density and angular momentum.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr

# Load data
P_range = np.linspace(0, 5, 100)
with np.load("./StarShapes/Results/P_scaling_radial.npz") as data:
    P_range = data["N_range"]
    N_dispersion = data["N_dispersion"]
    t_exec_dispersion = data["t_exec_dispersion"]
    result_dispersion = data["result_dispersion"]
    integral_dispersion = data["integral_dispersion"]

with np.load("./StarShapes/Results/P_scaling_angular_radial.npz") as data:
    P_range = data["N_range"]
    N_dispersion_angular = data["N_dispersion"]
    t_exec_dispersion_angular = data["t_exec_dispersion"]
    result_dispersion_angular = data["result_dispersion"]
    integral_dispersion_angular = data["integral_dispersion"]

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Sum the dispersion across nodes
ax[0].imshow(t_exec_dispersion, aspect="auto", origin="lower")

plt.show()

