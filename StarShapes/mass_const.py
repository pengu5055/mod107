"""
Calculate the mass of the weird star shape that was 
given in the assignment. It is assumed here that the density
is constant throughout the shape.
"""
import numpy as np
from necromancer import NumberNecromancer
import time

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
if True:
    N_range = np.logspace(2, 12, 50)
    # Result dispersion across nodes
    N_dispersion = []  # Slightly deviates due to floor division when chunking
    t_exec_dispersion = []
    result_dispersion = []
    for i, N in enumerate(N_range):
        nn = NumberNecromancer(condition, num_samples=N, num_dimensions=3, domain=[-1, 1])
        if nn.rank == 0:
            print(f"Running {i + 1} of {len(N_range)}")
        n_in, n_tot, t_exec, _ = nn.run()
        N_dispersion.append(n_tot)
        result_dispersion.append(n_in)
        t_exec_dispersion.append(t_exec)

    if nn.rank == 0:
        np.savez("./StarShapes/Results/N_scaling.npz", N_range=N_range, N_dispersion=N_dispersion, 
                t_exec_dispersion=t_exec_dispersion, result_dispersion=result_dispersion)
nn.burry()
quit()

# From here process only on rank 0
if nn.rank == 0:
    # Reduce the results
    n_in = np.sum(n_in)
    n_tot = np.sum(n_tot)

    # Calculate the mass
    mass = n_in / n_tot
    print(f"The mass/volume of the shape is {mass} units.")
    print(f"Computation took {t} s on {nn.size} slaves.")

# Let the dead rest again
nn.burry()

