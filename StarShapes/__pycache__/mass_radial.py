"""
Calculate mass of the star shape with a radially
dependent density:

r = r^p

Check what the parameter p does.
"""
import numpy as np
from necromancer import NumberNecromancer

# Default constants
P = 1

def condition(pairs):
    results = []
    for pair in pairs:
        x, y, z = pair
        check = np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1
        if check:
            np.sqrt(x**2 + y**2 + z**2)**P
        results.append(np.sqrt(np.abs(x)) + np.sqrt(np.abs(y)) + np.sqrt(np.abs(z)) <= 1)
    return np.array(results)


# --- Plot 1: Changes in p ---

