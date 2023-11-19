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
        self.client = Client()

        # Get the number of slaves
        comm = MPI.COMM_WORLD
        self.slaves = comm.Get_size()
        self.rank = comm.Get_rank()

        # Set the condition function
        self.condition_function = condition_function

        # Set the number of samples and dimensions
        self.num_samples = num_samples
        self.num_samples_node = int(num_samples / self.slaves)
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
                                        (self.num_samples_node, self.num_dimensions))
        
        return self.samples

    def _check_samples(self, samples: np.ndarray) -> int:
        """
        Check the samples against the condition function.

        Parameters
        ----------
        samples : np.ndarray
            The samples to check.

        Returns
        -------
        int
            The number of samples that satisfy the condition.
        """

        # Check the samples
        self.satisfied_samples = self.condition_function(samples)
        result = self.satisfied_samples.sum()

        return result

    def setup(self):
        """
        Setup the Monte Carlo integration.
        """

        # Generate the samples
        self._generate_samples()

        # Submit the task
        result_future = self.client.submit(self._check_samples, self.samples)

        # Wait for the task to finish
        self.satisfied = self.client.gather(result_future)

        return self.satisfied, self.num_samples

    def burry(self):
        """
        Burry the dead.
        """

        # Close the client
        self.client.close()


def init_necromancy(slaves: int):
    """
    Initialize the necromancy. Set's up the MPI processes.

    Parameters
    ----------
    integrator : NumberNecromancer
        The integrator to use.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Create the command
        command = f"mpiexec -np {slaves} python {__file__}"

        # Run the command
        subprocess.run(command, shell=True)

        # Exit 
        exit()
    else:
        pass

