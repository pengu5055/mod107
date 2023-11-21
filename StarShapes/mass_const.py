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

# Call buddy necromancer
nn = NumberNecromancer(condition, num_samples=10000000, num_dimensions=3, domain=[-1, 1])
# Run the necromancer
n_in, n_tot, t = nn.run()

# --- Plot 1: Scaling with N ---
if True:
    N_range = np.logspace(2, 20, 100)
    t_range = []
    result_dispersion = []
    for N in N_range:
        nn.num_samples = int(N)
        ti = time.time()
        n_in, n_tot = nn.run()
        tf = time.time() - ti
        t_range.append(tf)


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

