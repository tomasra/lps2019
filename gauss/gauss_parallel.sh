#!/bin/sh
mpiexec --oversubscribe -n $1 python3 gauss_parallel.py $2 $3 $4
