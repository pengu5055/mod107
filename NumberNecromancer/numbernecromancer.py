"""
The main class for the parallel Monte Carlo integrator.
Name inspired by the fact that I'd like a break and 
this is soul-sucking work.
"""
import numpy as np
from dask_mpi import initialize
from dask.distributed import Client
import dask.array as da
from typing import Union, Tuple, List, Callable
import subprocess
from mpi4py import MPI
import dask

class NumberNecromancer:
    def __init__(self,
                 condition_function: Callable,
                 num_samples: int,
                 num_dimensions: int,
                 domain: Tuple[float, float] = (0, 1),
                 
                 ) -> None:
        """
        The base class for the parallel Monte Carlo integrator.

        Parameters
        ----------
        condition_function : Callable
            The condition for which to check at each sample.
        num_samples : int
            The number of samples to take.
        num_dimensions : int
            The number of dimensions to sample.
        domain : Tuple[float, float], optional
            The domain to sample over, by default (0, 1).
        """

        # Initialize Dask
        initialize()
        self.client = Client(serializers=['pickle'])

        # Get the number of slaves
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size() 
        self.slaves = self.size - 2 
        self.rank = comm.Get_rank()

        # Set the condition function
        self.condition_function = condition_function

        # Set the number of samples and dimensions
        self.num_samples = num_samples
        self.num_dimensions = num_dimensions
        self.domain = domain
    
    def _generate_samples(self) -> np.ndarray:
        """
        Generate the samples to check. Override self.samples to change
        the distribution.

        Returns
        -------
        np.ndarray
            The samples to check.
        """
        # Generate the samples
        self.samples = da.random.uniform(self.domain[0], self.domain[1],
                                        (self.num_samples, self.num_dimensions),
                                        chunks=(self.num_samples // self.slaves, self.num_dimensions))
    
        
        return self.samples
    
    def setup(self):
        """
        Setup the Monte Carlo integration.
        """
        # Generate the samples
        samples = self._generate_samples()

        # Compute the condition
        satisfied = dask.delayed(self.condition_function)(samples)

        # Compute the number of satisfied samples
        self.satisfied = satisfied.sum()
        
        return self.satisfied, self.num_samples

    def compute(self):
        """
        Compute the Monte Carlo integration.
        """
        # Compute the condition
        self.satisfied = self.satisfied.compute()
        ## TODO: Currently computing is not parallelized. Only using 1 worker.

        return self.satisfied, self.num_samples

    def burry(self):
        """
        Burry the dead.
        """

        # Close the client
        self.client.close()
