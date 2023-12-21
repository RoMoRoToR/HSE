from numba import njit
from numba.openmp import openmp_context as openmp
from numba.openmp import omp_get_thread_num, omp_get_num_threads, omp_set_num_threads, omp_get_max_threads, omp_get_wtime
import numpy as np
import random

@njit
def getOmpTime():
    return omp_get_wtime()

@njit
def partition(arr, low, high):
    pivot = arr[(low + high) // 2]
    i = low
    j = high
    while True:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1

@njit
def quickSortPar(arr, low, high, max_d, d=0):
    if (low < high):
        i = low
        j = high
        pivot = arr[(i + j) // 2]
        while (1):
            while (arr[i] < pivot):
                i = i + 1
            while (arr[j] > pivot):
                j = j - 1
            if (i >= j):
                break
            arr[i], arr[j] = arr[j], arr[i]
            i = i + 1
            j = j - 1
        pi = j

        if (d < max_d):
            with openmp("task shared(arr)"):
                quickSortPar(arr, low, pi, max_d, d + 1)
            with openmp("task shared(arr)"):
                quickSortPar(arr, pi + 1, high, max_d, d + 1)
            with openmp("taskwait"):
                return
        else:
            quickSortPar(arr, low, pi, max_d, d + 1)
            quickSortPar(arr, pi + 1, high, max_d, d + 1)
@njit
def quickSortSeq(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSortSeq(arr, low, pi)
        quickSortSeq(arr, pi + 1, high)
@njit
def quickSortParHelp(arr, max_d):
    with openmp("parallel shared(arr)"):
        with openmp("single"):
            quickSortPar(arr, 0, len(arr) - 1, max_d)
@njit
def setNumThreads(n):
    omp_set_num_threads(n)

def createRandomArr(n):
    return np.random.randint(0, 100, n)

if __name__ == "__main__":
    sizes = [100000, 1000000, 10000000]
    threadsCounts = [1, 2, 4, 8, 16, 24]

    for N in sizes:
        print("Size:", N)
        arr_seq = createRandomArr(N)
        start_time_seq = getOmpTime()
        quickSortSeq(arr_seq, 0, N - 1)
        time_seq = getOmpTime() - start_time_seq
        print("Sequential QuickSort Time for Size {}: {} seconds".format(N, time_seq))

        for threads_n in threadsCounts:
            setNumThreads(threads_n)
            arr_par = createRandomArr(N)
            start_time_par = getOmpTime()
            quickSortParHelp(arr_par, 4)  # Assuming max_d is 4
            time_par = getOmpTime() - start_time_par
            speedup = time_seq / time_par if time_par > 0 else float('inf')
            print("Threads: {}, Parallel Time: {}, Speedup: {}".format(threads_n, time_par, speedup))
