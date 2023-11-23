"""
The .py file to spawn via subprocess for the purpose of profiling the necromancer.
This is called from node_scaling.py.
"""
import numpy as np
import pandas as pd
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
    # Pad the results with zeros to make all arrays the same length
    # This is done to make it easier to process the data
    max_len = 152
    n_in = np.pad(n_in, (0, max_len - len(n_in)), mode="constant")
    n_tot = np.pad(n_tot, (0, max_len - len(n_tot)), mode="constant")
    t_exec = np.pad(t_exec, (0, max_len - len(t_exec)), mode="constant")

    # Append the results to h5 file
    store = pd.HDFStore('./StarShapes/Results/node_scaling.h5')
    df = pd.DataFrame({"n_in": n_in, "n_tot": n_tot, "t_exec": t_exec})
    store.append("node_scaling", df, format="table", data_columns=True, complevel=9)
    store.close()

# Let the dead rest again
nn.burry()
