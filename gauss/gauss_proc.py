#!/usr/bin/env python
import csv
import sys
import time
import numpy as np
from itertools import cycle
from multiprocessing import Process, Barrier, Queue, sharedctypes


# Problem size (matrix dimension)
N = int(sys.argv[1])

# Block/cyclic parallelization mode
MODE = sys.argv[2]

# Number of repeats for averaging times
REPEATS = int(sys.argv[3])

# Number of processes
PROC_COUNT = int(sys.argv[4])
MAIN_PROC = 0

data_shape = (N, N + 1)
x_calc_shape = (N,)

barrier = Barrier(PROC_COUNT)
x_calc_queue = Queue()

def worker_func(data, tasks, x_calc, proc_id):
    data = np.frombuffer(data, dtype=np.float64).reshape(data_shape)
    tasks = np.frombuffer(tasks, dtype=np.int32).reshape(data_shape)
    # x_calc = np.frombuffer(x_calc, dtype=np.float64).reshape(x_calc_shape)
    x_calc = np.zeros(x_calc_shape, dtype=np.float64)

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
        barrier.wait()
        for j in np.where(tasks[k] == proc_id)[0]:
            _do_task(k, j)

    # Backward stage
    if proc_id == MAIN_PROC:
        for k in range(N - 1, -1, -1):
            x_calc[k] = data[k, N] / data[k, k]
            rows = slice(0, k)
            data[rows, N] -= data[rows, k] * x_calc[k]
        # x_calc_queue.put(x_calc)

times = []
for run_id in range(REPEATS):
    # Prepare data
    np.random.seed(1000 + run_id)
    matrix = np.random.rand(N, N).astype(np.float64)
    matrix = matrix + np.diag(np.sum(matrix, axis=1))
    vector = np.random.rand(N)

    x_true = np.linalg.solve(matrix, vector)
    x_calc = np.zeros(vector.shape, dtype=np.float64)

    # Fill buffer
    data = np.concatenate((matrix, vector.reshape(-1, 1)), axis=1).astype(np.float64)

    # Block/cyclic distribution of tasks
    # -1 represents no task, other integers - assigned process IDs
    tasks = np.full(data_shape, -1, dtype=np.int32)
    if MODE == 'block':
        for pid in range(PROC_COUNT):
            col_start = int(np.round((N / PROC_COUNT) * pid)) + 1
            col_end = int(np.round((N / PROC_COUNT) * pid + (N / PROC_COUNT))) + 1
            for col in range(col_start, col_end):
                for row in range(0, col):
                    tasks[row, col] = pid
    elif MODE == 'cyclic':
        pids = cycle(range(PROC_COUNT))
        for k in range(1, N + 1):
            pid = next(pids)
            tasks[0:(k), k] = pid
    else:
        exit()

    # Share data among processes
    sc_data = sharedctypes.RawArray('d', data.reshape(-1))
    sc_tasks = sharedctypes.RawArray('i', tasks.reshape(-1))
    sc_x_calc = sharedctypes.RawArray('d', x_calc.reshape(-1))
    
    start_time = time.time()
    processes = []
    for proc_id in range(PROC_COUNT):
        worker_args = (sc_data, sc_tasks, sc_x_calc, proc_id)
        process = Process(target=worker_func, args=worker_args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    end_time = time.time()
    duration = end_time - start_time
    times.append(duration)

    # x_calc = x_calc_queue.get()
    # print(np.allclose(x_true, x_calc, atol=np.finfo(np.float32).eps))

# Save results
with open('gauss_proc.csv', 'a') as fp:
    writer = csv.writer(fp)
    writer.writerow([N, MODE, PROC_COUNT, np.mean(times)])
