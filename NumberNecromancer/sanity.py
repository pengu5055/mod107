"""
Check my sanity.
"""
import time
import numpy as np
from numbernecromancer import NumberNecromancer
import dask.array as da
from time import sleep

def condition(pairs):
    print("\n\n CALLED \n\n\n")
    results = []
    for pair in pairs:
        x, y = pair
        results.append(x ** 2 + y ** 2 <= 1)
    return np.array(results)

# Create the necromancer
nn = NumberNecromancer(condition, num_samples=10000000, num_dimensions=2, domain=[0, 1])
nn.setup()
ti = time.time()
n_in, n_tot = nn.compute()
tf = time.time() - ti

print("\n-------RESULTS-------\n")
print(f"Computation took: {tf} s on {nn.slaves} slaves.")
print(f"n_in: {n_in}, n_tot: {n_tot}")
print(f"By our logic pi is {4 * n_in / n_tot}")
print("\n---------------------\n")

nn.burry()
