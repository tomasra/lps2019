#!/bin/sh
#SBATCH -p short
#SBATCH -C alpha
mpirun --pernode -n $1 python3 gauss_mpi.py $2 $3 $4
