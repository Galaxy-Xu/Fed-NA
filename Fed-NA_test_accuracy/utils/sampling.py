#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def openSamplingFile(filepath):
    file = open(filepath)
    dict_users = {}
    index = 0
    while True:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        temp = []
        line = line[0:len(line)-2]
        line = line.split(',')
        # print(line)
        for cur in line:
            temp.append(int(cur))
        dict_users[index] = set(temp)
        index += 1
        if not line:
            break
        pass
    file.close()
    return dict_users


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    filePath = '../data/mnist_iid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_items = int(len(dataset) / num_users)
        #num_items = int(len(dataset) / 600)
        all_idxs = [i for i in range(len(dataset))] # [0,1,2,3, ..., 59999]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False)) # {[0,1,2, ..., 598, 599]}
            all_idxs = list(set(all_idxs) - dict_users[i]) # {[600, .., 59999]}
        # for i in range(0,20):
        #     dict_users[i] = set(np.random.choice(all_idxs, num_items*10, replace=False))
        #     all_idxs = list(set(all_idxs) - dict_users[i])
        # for i in range(20,60):
        #     dict_users[i] = set(np.random.choice(all_idxs, num_items*5, replace=False))
        #     all_idxs = list(set(all_idxs) - dict_users[i])
        # for i in range(60,120):
        #     dict_users[i] = set(np.random.choice(all_idxs, num_items*3, replace=False))
        #     all_idxs = list(set(all_idxs) - dict_users[i])
        # for i in range(120,140):
        #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        #     all_idxs = list(set(all_idxs) - dict_users[i])
    if dict_users == {}:
        return "Error"
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    filePath = '../data/mnist_noniid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2)) # 200 , 300
        idx_shard = [i for i in range(num_shards)] # [0, 1, ...,199]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)} # {0: array, 1: array, ..., 99:array}
        idxs = np.arange(num_shards * num_imgs)#num_shards:200 , num_imgs:300 # np.arange(len(dataset))
        labels = dataset.train_labels.numpy()

        # sort `index` by` `labels`
        idxs_labels = np.vstack((idxs, labels))
        #[[0 1 2 ...                 59999]
        # [5 4 0 ..1 7..2 4..3 3.....7 1]]
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] #将标签排序
        idxs = idxs_labels[0, :]#将标签顺序排好的数据序号再排序
        #[[0 1 2 ...                 59999]
        # [0 0 0 ..1 1..2 2..3 3.....9 9]]


        # divide and assign
        for i in range(num_users):
            #rand_set = set(np.random.choice(idx_shard, 2, replace=False))#replace=True，有可能会出现重复的元素   最极端情况不适用random
            rand_set = set(idx_shard[0:2])
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)#拼接两个idx_shard的num_imgs数量
    if dict_users == {}:
        return "Error"
    return dict_users # {0: array, 1: array, ..., 99:array}


def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    filePath = '../data/fashion_iid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_items = int(len(dataset) / num_users)
        all_idxs = [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
    if dict_users == {}:
        return "Error"
    return dict_users


def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    filePath = '../data/fashion_noniid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs = np.arange(num_shards * num_imgs)
        labels = dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    if dict_users == {}:
        return "Error"
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    filePath = '../data/cifar_iid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
    if dict_users == {}:
        return "Error"
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    filePath = '../data/cifar_noniid_{}clients.dat'.format(num_users)
    dict_users = {}
    try:
        dict_users = openSamplingFile(filePath)
    except FileNotFoundError:
        num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs = np.arange(num_shards * num_imgs)
        # labels = dataset.train_labels.numpy()
        labels = np.array(dataset.targets)
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            #rand_set = set(idx_shard[0:2])
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    if dict_users == {}:
        return "Error"
    return dict_users


if __name__ == '__main__':
    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST('../data/fashion-mnist-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    print(fashion_iid(dataset_train, 1000)[0])


