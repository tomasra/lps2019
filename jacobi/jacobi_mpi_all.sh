#!/bin/bash
REPEATS=5
for PROC_COUNT in 1
do
    for N in 1000 2000 3000 4000 5000
    do
        sbatch -p short -C alpha -N $PROC_COUNT jacobi_mpi.sh $PROC_COUNT $N $REPEATS
    done
done