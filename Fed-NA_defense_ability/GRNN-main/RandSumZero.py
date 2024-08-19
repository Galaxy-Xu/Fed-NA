import random
import numpy as np


def rand_sum_zero(n):
    upper = 1
    lower = -1
    arr = np.zeros(n)
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

# def creat_noise_1():
#     noise = np.zeros([10, 450])
#     for j in range(450):
#         arr = rand_sum_zero(10)
#         for k in range(10):
#             noise[k, j] = arr[k]
#     return noise


def creat_noise_1():
    noise = np.zeros([10, 450])

    for j in range(450):
        arr = rand_sum_zero(10)
        arr_abs = abs(arr)
        indices_descending = np.argsort(arr_abs)
        arr_final = arr[indices_descending]
        for k in range(10):
            noise[k, j] = arr_final[k]

    return noise



def creat_noise_2():
    noise = np.zeros([10, 2400])

    for j in range(2400):
        arr = rand_sum_zero(10)
        for k in range(10):
            noise[k, j] = arr[k]

    return noise

# print("init ok, waiting for connect")
# print("received model from 192.168.42.37:8002")
# print("server testing ...")
# print("test acc: 0.11")
# print("send to client 192.168.42.103:8002")


# print("received model from server ")
# print("begin train ")
# print("upload ok, waiting for the next training...")