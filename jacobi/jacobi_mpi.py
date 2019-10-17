import csv
import sys
import time
import numpy as np
from itertools import cycle
from mpi4py import MPI


MAX_ITER_COUNT = 10000
EPSILON = 1e-5

# Problem size (matrix dimension)
N = int(sys.argv[1])

# Number of repeats for averaging times
REPEATS = int(sys.argv[2])

comm = MPI.COMM_WORLD
proc_id = comm.Get_rank()
proc_count = comm.Get_size()
MAIN_PROC = 0

proc_slices = [slice(
    int(np.round((N / proc_count) * pid)),
    int(np.round((N / proc_count) * pid + (N / proc_count))),)
    for pid in range(proc_count)]

proc_slice = proc_slices[proc_id]


def jacobi(a, b, eps=1e-5):
    n = a.shape[0]
    m = a.copy()
    
    # Matrix transformation
    diag_coefs = np.diagonal(a)
    for i in range(n):
        m[i] = m[i] / diag_coefs[i] * -1
    m[np.diag_indices(n, 2)] = b / diag_coefs
    
    # Iteration loop
    iters = 0
    x = np.zeros(b.shape)
    for itr in range(MAX_ITER_COUNT):
        x_prev = x.copy()
        # Single iteration
        for i in range(proc_slice.start, proc_slice.stop):
            # Reuse the same array, just set 1 for current x[i]
            xc = x_prev[i]
            x_prev[i] = 1
            x[i] = np.dot(m[i], x_prev)
            x_prev[i] = xc

        if proc_count > 1:
            x_recv = np.empty(x.shape)
            if proc_id == 0:
                # First process in the chain
                comm.Send([x, MPI.FLOAT], dest=proc_id + 1)
                comm.Recv([x_recv, MPI.FLOAT], source=proc_id + 1)
                x_recv[proc_slice] = x[proc_slice]
                x = x_recv
            elif proc_id == proc_count - 1:
                # Last process in the chain
                comm.Recv([x_recv, MPI.FLOAT], source=proc_id - 1)
                x[:proc_slice.start] = x_recv[:proc_slice.start]
                comm.Send([x, MPI.FLOAT], dest=proc_id - 1)
            else:
                # Some middle process
                comm.Recv([x_recv, MPI.FLOAT], source=proc_id - 1)
                x_recv[proc_slice] = x[proc_slice]
                x = x_recv
                comm.Send([x, MPI.FLOAT], dest=proc_id + 1)
                comm.Recv([x_recv, MPI.FLOAT], source=proc_id + 1)
                remaining = slice(proc_slices[proc_id + 1].start)
                x[remaining] = x_recv[remaining]
                comm.Send([x, MPI.FLOAT], dest=proc_id - 1)

        iters += 1

        res = np.linalg.norm(x - x_prev, np.inf)
        # print(proc_id, iters, res)
        if res < eps:
            break
    else:
        raise RuntimeError('Did not converge in ', MAX_ITER_COUNT, ' iterations')
    return x, iters



# times, iter_counts = [], []
for run_id in range(REPEATS):
    # Prepare data
    np.random.seed(1000 + run_id)
    matrix = np.random.rand(N, N)
    matrix = matrix + np.diag(np.sum(matrix, axis=1))
    vector = np.random.rand(N)
    # x_true = np.linalg.solve(matrix, vector)

    start_time = time.time()
    x_calc, iters = jacobi(matrix, vector, eps=EPSILON)
    end_time = time.time()
    duration = end_time - start_time
    # print(end_time - start_time)
    # times.append(end_time - start_time)
    # iter_counts.append(iters)
    # print(iters)

    if proc_id == 0:
        # Save results
        with open('jacobi_mpi.csv', 'a') as fp:
            writer = csv.writer(fp)
            writer.writerow([N, proc_count, run_id, duration, iters])

MPI.Finalize()
