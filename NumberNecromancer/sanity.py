"""
Check my sanity.
"""
import time
import numpy as np
from necromancer import NumberNecromancer

def condition(pairs):
    results = []
    for pair in pairs:
        x, y = pair
        results.append(x ** 2 + y ** 2 <= 1)
    return np.array(results)

# Create the necromancer
nn = NumberNecromancer(condition, num_samples=10000000, num_dimensions=2, domain=[0, 1])
n_in, n_tot, t_exec, res = nn.run()
n_in = np.sum(n_in)
n_tot = np.sum(n_tot)

if nn.rank == 0:
    print("\n-------RESULTS-------")  
    print(f"n_in: {n_in}, n_tot: {n_tot}")
    print(f"With res I got {res}")
    print(f"By our logic pi is {4 * n_in / n_tot}")
    print(f"Computation took: {t_exec} s on {nn.size} slaves.")
    print("\n---------------------")

nn.burry()
