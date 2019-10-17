#!/bin/sh
mpiexec --pernode -n $1 python3 jacobi_mpi.py $2 $3
