#!/usr/bin/env python
import csv
import sys
import time
import numpy as np

REPEATS = 5
N = int(sys.argv[1])

def gauss_seq2(a, b):
    n = a.shape[0]
    m = np.concatenate((a, b.reshape(-1, 1)), axis=1)
    x = np.zeros(b.shape)
    
    # Forward
    # k - row being subtracted
    # j - current column
    # i - current row
    for k in range(0, n):
        for j in range(k + 1, n + 1):
            t = m[k, j] / m[k, k]          
#             for i in range(k + 1, n):
#                 m[i, j] -= m[i, k] * t
            rows = slice(k + 1, n)
            m[rows, j] -= m[rows, k] * t
        
    # Backward
    for k in range(n - 1, -1, -1):
        x[k] = m[k, n] / m[k, k]
#         for j in range(k - 1, -1, -1):
#             m[j, n] -= m[j, k] * x[k]
        rows = slice(0, k)
        m[rows, n] -= m[rows, k] * x[k]
    
    return x

if __name__ == '__main__':
    times = []
    for run_id in range(REPEATS):
        np.random.seed(1001)
        matrix = np.random.rand(N, N)
        matrix = matrix + np.diag(np.sum(matrix, axis=1))
        vector = np.random.rand(N)
        x_true = np.linalg.solve(matrix, vector)

        start_time = time.time()
        x_calc = gauss_seq2(matrix, vector)
        end_time = time.time()

    # Save results
    with open('gauss.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow([N, mode, proc_count, np.mean(times)])

    # print(end_time - start_time)
    # print(np.allclose(x_true, x_calc, atol=np.finfo(np.float32).eps))
