import random
import numpy as np

def rand_sum_zero(n):
    upper = 1
    lower = -1
    arr = np.zeros(n)  # A zero array of length n.
    arr[0] = random.uniform(lower, upper)
    asum = arr[0]
    for i in range(1, n - 1):
        if asum > 0:
            upper = 1 - asum
            lower = -1
        else:
            lower = -1 - asum
            upper = 1
        arr[i] = random.uniform(lower, upper)
        asum = asum + arr[i]
    arr[n - 1] = -asum

    return arr

def creat_noise(m):
    noise = np.zeros([m, 16384])

    for j in range(16384):
        arr = rand_sum_zero(m)
        arr_abs = abs(arr)
        indices_descending = np.argsort(arr_abs)
        arr_final = arr[indices_descending]
        for k in range(m):
            noise[k, j] = arr_final[k]

    noise = noise.reshape(m,64,256)

    return noise





