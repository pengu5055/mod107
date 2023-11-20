"""
Test in development code.
Try to find the value of pi using a Monte Carlo method.
"""
import numpy as np
from numbernecromancer import NumberNecromancer

# condition = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2 <= 1
# condition = lambda x: print(x[:, 0], x[:, 1])

def condition(pairs):
    results = []
    for pair in pairs:
        x, y = pair
        results.append(x ** 2 + y ** 2 <= 1)
    return np.array(results)

# Create the necromancer
nn = NumberNecromancer(condition, num_samples=1000, num_dimensions=2, domain=[0, 1])

# Run the necromancer
# nn.setup()
# n_in, n_tot = nn.compute()

# Print the result
# print(4 * n_in / n_tot)

# Close the client
nn.burry()