#!/bin/bash
REPEATS=5
for PROC_COUNT in 1 2 3 4 5 6 7 8 9 10
do
    for N in 1000 2000 3000 4000 5000
    do
        for MODE in block cyclic
        do
            python3 gauss_proc.py $N $MODE $REPEATS $PROC_COUNT
        done
    done
done