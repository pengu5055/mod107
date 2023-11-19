"""
Try out Dask
"""
import numpy as np
from dask_mpi import initialize
from dask.distributed import Client
import dask.array as da
import os
from time import sleep

# Initialize Dask
initialize()
client = Client()

# Create a random array
a = da.random.random((1000, 1000), chunks=(100, 100))
a = np.sin(a)

sleep(10)

# Compute the mean
print(a.mean().compute())
client.close()