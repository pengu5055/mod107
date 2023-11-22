!# /bin/bash

# This script iterates over all the nodes

for i in {1..152}
do
    echo "Running node $i"
    mpiexec -np $i python -m mpi4py ./StarShapes/to_spawn.py
done