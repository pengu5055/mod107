"""
The .py file to spawn via subprocess for the purpose of profiling the necromancer.
This is called from node_scaling.py.
"""
import numpy as np
from necromancer import NumberNecromancer
import os
# Let's start of with 3 dimensions and complicate it if time allows
# Maybe display the shape in 3D just for fun
def condition(pairs):
    results = []
    for pair in pairs:
        x, y, z = pair
        results.append(np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1)
    return np.array(results)

nn = NumberNecromancer(condition, num_samples=1e7, num_dimensions=3, domain=[-1, 1])
n_in, n_tot, t_exec, _ = nn.run()

# From here process only on rank 0
if nn.rank == 0:
    # Reduce the results
    with open(f"./StarShapes/Results/dump.txt", "a") as f:
        f.write(f"{n_in}\t{n_tot}\t{t_exec}\n")

# Let the dead rest again
nn.burry()
