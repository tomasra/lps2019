#!/usr/bin/env python
import sys
import time
import csv
import numpy as np

REPEATS = 5
MAX_ITER_COUNT = 10000
EPSILON = 1e-5
N = int(sys.argv[1])


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
        for i in range(n):
            # Reuse the same array, just set 1 for current x[i]
            xc = x_prev[i]
            x_prev[i] = 1
            x[i] = np.dot(m[i], x_prev)
            x_prev[i] = xc
            
        iters += 1
        if np.linalg.norm(x - x_prev, np.inf) < eps:
            break
    else:
        raise RuntimeError(f'Did not converge in {MAX_ITER_COUNT} iterations')
    return x, iters


if __name__ == '__main__':
    times, iter_counts = [], []
    for run_id in range(REPEATS):
        np.random.seed(1001)
        matrix = np.random.rand(N, N)
        matrix = matrix + np.diag(np.sum(matrix, axis=1))
        vector = np.random.rand(N)
        x_true = np.linalg.solve(matrix, vector)

        start_time = time.time()
        x_calc, iters = jacobi(matrix, vector, eps=EPSILON)
        end_time = time.time()
        # print(end_time - start_time)
        # print(iters)
        times.append(end_time - start_time)
        iter_counts.append(iters)

    # Save results
    with open('jacobi_seq.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow([N, 1, np.mean(times), np.mean(iter_counts)])
