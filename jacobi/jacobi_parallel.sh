#!/bin/sh
mpiexec --oversubscribe -n $1 python3 jacobi_parallel.py $2 $3
