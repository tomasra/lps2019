import csv
import sys
import time
import numpy as np
from itertools import cycle
from mpi4py import MPI

# Problem size (matrix dimension)
N = int(sys.argv[1])

# Block/cyclic parallelization mode
mode = sys.argv[2]

# Number of repeats for averaging times
REPEATS = int(sys.argv[3])

comm = MPI.COMM_WORLD
proc_id = comm.Get_rank()
proc_count = comm.Get_size()
MAIN_PROC = 0

print(proc_id, ' starting')

times = []
for run_id in range(REPEATS):
    # Prepare data
    np.random.seed(1000 + run_id)
    matrix = np.random.rand(N, N)
    matrix = matrix + np.diag(np.sum(matrix, axis=1))
    vector = np.random.rand(N)

    x_true = np.linalg.solve(matrix, vector)
    x_calc = np.zeros(vector.shape)
    data = np.concatenate((matrix, vector.reshape(-1, 1)), axis=1)

    # Block/cyclic distribution of tasks
    # -1 represents no task, other integers - assigned process IDs
    tasks = np.full(data.shape, -1, dtype=np.int8)
    if mode == 'block':
        for pid in range(proc_count):
            col_start = int(np.round((N / proc_count) * pid)) + 1
            col_end = int(np.round((N / proc_count) * pid + (N / proc_count))) + 1
            for col in range(col_start, col_end):
                for row in range(0, col):
                    tasks[row, col] = pid
    elif mode == 'cyclic':
        pids = cycle(range(proc_count))
        for k in range(1, N + 1):
            pid = next(pids)
            tasks[0:(k), k] = pid
    else:
        exit()

    def _do_task(k, j):
        t = data[k, j] / data[k, k]
        rows = slice(k + 1, N)
        data[rows, j] -= data[rows, k] * t
        # Mark as completed
        tasks[k, j] = -1

    start_time = time.time()

    # Forward stage
    for k in range(0, N):
        task_proc = tasks[k, k + 1]
        if task_proc == proc_id:
            _do_task(k, k + 1)

            # Distribute to other processes
        #     for pid in range(proc_count):
        #         if pid == proc_id:
        #             continue
        #         buf = data[:, k + 1].copy()
        #         comm.Isend([buf, MPI.FLOAT], dest=pid, tag=(k + 1))
        # else:
        #     buf = np.empty(data[:, k + 1].shape)
        #     comm.Recv(buf, source=tasks[k, k + 1], tag=(k + 1))
        #     data[:, k + 1] = buf

        buf = data[:, k + 1].copy()
        comm.Bcast(buf, root=task_proc)
        data[:, k + 1] = buf

        for j in np.where(tasks[k] == proc_id)[0]:
            _do_task(k, j)


    # Backward stage
    if proc_id == MAIN_PROC:
        for k in range(N - 1, -1, -1):
            x_calc[k] = data[k, N] / data[k, k]
            rows = slice(0, k)
            data[rows, N] -= data[rows, k] * x_calc[k]

        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
        # print(end_time - start_time)
        print(np.allclose(x_true, x_calc, atol=np.finfo(np.float32).eps))


if proc_id == MAIN_PROC:
    # Save results
    with open('gauss_mpi.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow([N, mode, proc_count, np.mean(times)])
