#! /bin/bash

# Create a sort of queue for tasks that need to run
# while I go to get a bare minimum of sleep

# This script will be run from the root directory of the project
# and will be run with the command:
# ./queue.sh

mpiexec -np 152 --hostfile hosts python3 -m mpi4py ./GammaBirth/unit_sphere.py; \
mpiexec -np 152 --hostfile hosts python3 -m mpi4py ./NeutronReflector/linear_motion.py; \
mpiexec -np 152 --hostfile hosts python3 -m mpi4py ./NeutronReflector/isotropic_motion.py
