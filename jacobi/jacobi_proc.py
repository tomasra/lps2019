#!/usr/bin/env python
import csv
import sys
import time
import numpy as np
from itertools import cycle
from multiprocessing import Process, Barrier, Lock, Queue, sharedctypes


MAX_ITER_COUNT = 100000
EPSILON = 1e-3

# Problem size (matrix dimension)
N = int(sys.argv[1])

# Number of repeats for averaging times
REPEATS = int(sys.argv[2])
PROC_COUNT = int(sys.argv[3])
MAIN_PROC = 0

proc_slices = [slice(
    int(np.round((N / PROC_COUNT) * pid)),
    int(np.round((N / PROC_COUNT) * pid + (N / PROC_COUNT))),)
    for pid in range(PROC_COUNT)]

barrier = Barrier(PROC_COUNT)
x_lock = Lock()
q_iters = Queue()


def worker_func(matrix, vector, x_calc_, proc_id):
    matrix = np.frombuffer(matrix, dtype=np.float64).reshape((N, N))
    vector = np.frombuffer(vector, dtype=np.float64).reshape((N,))
    x_calc = np.frombuffer(x_calc_, dtype=np.float64).reshape((N,))
    proc_slice = proc_slices[proc_id]

    # Iteration loop
    iters = 0
    # x = np.zeros(vector.shape)
    for itr in range(MAX_ITER_COUNT):
        # print('before', proc_id, iters, x_calc)

        x_prev = x_calc.copy()
        barrier.wait()
        # x_prev = x_calc[:]
        # Single iteration
        for i in range(proc_slice.start, proc_slice.stop):
            # Reuse the same array, just set 1 for current x[i]
            xc = x_prev[i]
            x_prev[i] = 1
            x_calc[i] = np.dot(matrix[i], x_prev)
            x_prev[i] = xc


        # print('before barrier', proc_id, iters, x_calc)
        barrier.wait()
        # x_calc_copy = x_calc.copy()
        # print('after barrier', proc_id, iters, x_prev, x_calc_copy)
        # barrier.wait()

        # print('after', proc_id, iters, x_calc)

        res = np.linalg.norm(x_calc - x_prev, np.inf)
        # print('residual', proc_id, iters, res)
        if res < EPSILON:
            if proc_id == MAIN_PROC:
                q_iters.put(iters)
            break
        iters += 1
    else:
        raise RuntimeError(f'Did not converge in {MAX_ITER_COUNT} iterations')


# times, iter_counts = [], []
for run_id in range(REPEATS):
    # Prepare data
    np.random.seed(1000 + run_id)
    matrix = np.random.rand(N, N).astype(np.float64)
    matrix = matrix + np.diag(np.sum(matrix, axis=1))
    vector = np.random.rand(N).astype(np.float64)
    
    # Initial solution: all zeros
    x_calc = np.zeros(vector.shape, dtype=np.float64)
    # x_true = np.linalg.solve(matrix, vector)

    start_time = time.time()

    # Prepare cofficients
    diag_coefs = np.diagonal(matrix)
    for i in range(N):
        matrix[i] = matrix[i] / diag_coefs[i] * -1
    matrix[np.diag_indices(N, 2)] = vector / diag_coefs

    # Prepare to share among processes
    sc_matrix = sharedctypes.RawArray('d', matrix.reshape(-1))
    sc_vector = sharedctypes.RawArray('d', vector.reshape(-1))
    sc_x_calc = sharedctypes.RawArray('d', x_calc.reshape(-1))

    processes = []
    for proc_id in range(PROC_COUNT):
        worker_args = (sc_matrix, sc_vector, sc_x_calc, proc_id)
        process = Process(target=worker_func, args=worker_args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    end_time = time.time()
    duration = end_time - start_time
    # times.append(duration)
    # print(duration)

    # Save results
    iters = q_iters.get()
    # print(duration, iters)
    with open('jacobi_proc.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow([N, PROC_COUNT, run_id, duration, iters])
