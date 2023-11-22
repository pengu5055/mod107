"""
Check if results are nonzero.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable as pl
import cmasher as cmr


with np.load("./StarShapes/Results/P_scaling_radial.npz") as data:
    P_range = data["N_range"]
    print(P_range)
    N_dispersion = data["N_dispersion"]
    print(N_dispersion)
    t_exec_dispersion = data["t_exec_dispersion"]
    print(t_exec_dispersion)
    result_dispersion = data["result_dispersion"]
    print(result_dispersion)
    integral_dispersion = data["integral_dispersion"]
    print(integral_dispersion)

with np.load("./StarShapes/Results/P_scaling_angular_radial.npz") as data:
    P_range = data["N_range"]
    print(P_range)
    N_dispersion_angular = data["N_dispersion"]
    print(N_dispersion_angular)
    t_exec_dispersion_angular = data["t_exec_dispersion"]
    print(t_exec_dispersion_angular)
    result_dispersion_angular = data["result_dispersion"]
    print(result_dispersion_angular)
    integral_dispersion_angular = data["integral_dispersion"]
    print(integral_dispersion_angular)
