"""
A MPI only version of the NumberNecromancer class. This class is used to
perform parallel Monte Carlo integration. Dask seems promising, but it
isn't working yet. The concept is great but it is hard to implement.
"""
from mpi4py import MPI
import numpy as np
from typing import Union, Tuple, List, Callable
import socket
import time

class NumberNecromancer:
    def __init__(self,
                 condition_function: Callable,
                 num_samples: int,
                 num_dimensions: int,
                 domain: Tuple[float, float] = (0, 1),
                 quiet_slaves: bool = False        
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
        quiet_slaves : bool, optional
            Whether to print information from the slaves, by default False.
        """
        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size() 
        self.rank = self.comm.Get_rank()

        # Set the condition function
        self.condition_function = condition_function

        # Set the number of samples and dimensions
        self.num_samples = num_samples
        self.sample_chunk = int(num_samples / self.size)
        self.num_dimensions = num_dimensions
        self.domain = domain

        # Initialize container for samples
        self.samples = None

        self.quiet_slaves = quiet_slaves

        if not self.quiet_slaves:
            print(f"Rank {self.rank} of {self.size} risen from the dead on {socket.gethostname()}")
    
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
        self.samples = np.random.uniform(self.domain[0], self.domain[1], 
                                    size=(self.sample_chunk, self.num_dimensions))
        return self.samples
    
    def _check_condition(self) -> np.ndarray:
        """
        Check the condition at each sample.

        Returns
        -------
        np.ndarray
            The results of the condition check.
        """
        # Check the condition
        return self.condition_function(self.samples)
    
    def run(self):
        """
        Perform the Monte Carlo integration.
        """
        t_init = time.time()
        # Generate the samples
        self._generate_samples()

        # Check the condition
        results = self._check_condition()

        # Reduce the results
        n_in = np.count_nonzero(results)
        result = np.sum(results)
        # print(f"Rank {self.rank} of {self.size} found {n_in} in samples")

        # Get execution time
        t_exec = time.time() - t_init

        # Gather the total number of in samples
        n_in_tot = self.comm.gather(n_in, root=0)

        # Gather results
        results_tot = self.comm.gather(result, root=0)
        
        # Gather the execution time
        t_exec_tot = self.comm.gather(t_exec, root=0)

        # Reduce the number of samples
        n_tot = np.repeat(np.array(self.sample_chunk), self.size)

        if self.rank == 0:
            return n_in_tot, n_tot, t_exec_tot, results_tot
        else:
            return None, None, t_exec, None
    
    def burry(self):
        """
        Burry the dead.
        """
        if not self.quiet_slaves:
            print(f"Rank {self.rank} of {self.size} burried on {socket.gethostname()}")

        # Close the client
        MPI.Finalize()
