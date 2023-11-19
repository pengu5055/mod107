"""
Test in development code.
Try to find the value of pi using a Monte Carlo method.
"""
from numbernecromancer import NumberNecromancer, init_necromancy

condition = lambda x, y: x ** 2 + y ** 2 <= 1

# Create the necromancer
nn = NumberNecromancer(condition, num_samples=1000, num_dimensions=2, domain=[0, 1])

# Run the necromancer
n_in, n_tot = nn.setup()

# Print the result
print(4 * n_in / n_tot)