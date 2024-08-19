#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
from datetime import datetime

import matplotlib
from torch.utils.data import DataLoader

from models import Nets

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from utils.flatten import flatten_gradients
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from Rand_Sum_Zero import creat_noise



if __name__ == '__main__':
    start_time = time.time()

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.Resize((32,32)),
                                 transforms.Grayscale(num_output_channels=3),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR100('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist-mnist'):
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden1=256, dim_hidden2=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    local_acc_test = []
    acc_test = []
    loss_test = []
    learning_rate = [args.lr for i in range(args.num_users)]

    data_loader = DataLoader(dataset_train, batch_size=args.bs)

    # The leakage risk of each client's ten layers of gradient information.
    Layer_risk = np.zeros((args.num_users,6))

    # Layer_risk / sum. Sum each row of data, representing the total weight of each layer.
    Layer_risk_weight = np.zeros((1, args.num_users))



    for iter in range(args.epochs):
        w_locals, loss_locals = [], []
        #w_lastlocals = w_locals

        # Used to store the sum of the leakage risks of the selected clients.
        Layer_sum = np.zeros((1,10))

        m = max(int(args.frac * args.num_users), 1)

        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        idxs_users = np.random.choice(np.arange(args.num_users), m, replace=False)

        # Layer_risk / sum Then sum each row of data to represent the total weight of each layer.
        Layer_risk_weight_local = np.zeros((1, m))


        for i in range(m):

            # If the selected clients have not been chosen before (`Layer_sum=0`), then the weights are not calculated since they are all zero.
            if( np.array_equal(Layer_sum,np.zeros((1,10)))):
                break
            else:
                Layer_risk_weight[0,[idxs_users[i]]] = (Layer_risk[idxs_users[i]] / Layer_sum).sum(axis=1)
                Layer_risk_weight_local[0, i] = (Layer_risk[idxs_users[i]] / Layer_sum).sum(axis=1)

        # Sort `weight` and retrieve the sorted indices.
        sorted_indices = np.argsort(Layer_risk_weight_local)[::-1]

        # Reorder `idxs_users` in ascending order using the sorted indices, ensuring that the client with the largest weight is placed last.
        idxs_user = np.array(idxs_users)
        idxs_user = idxs_user[sorted_indices]

        # Generate zero-sum noise based on the clients' weights.
        noise_convolution = creat_noise(m)
        noise_add = torch.from_numpy(noise_convolution).to(args.device)
        target = 0

        local_models = []

        for idx in idxs_users:
        # for idx in idxs_user[0]:
            args.lr = learning_rate[idx]
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, curLR = local.train(w_glob,net=copy.deepcopy(net_glob).to(args.device))
            learning_rate[idx] = curLR
            local_models.append(copy.deepcopy(w))  # Save the local model

        #************************** Soteria ***************************
            # feature_fc1_graph = Nets.extract_feature()
            #
            # deviation_f1_target = torch.zeros_like(feature_fc1_graph)
            # deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
            # for f in range(deviation_f1_x_norm.size(1)):
            #     deviation_f1_target[:, f] = 1
            #     feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
            #     deviation_f1_x = ground_truth.grad.data
            #
            #     deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
            #                 feature_fc1_graph.data[:, f] + 0.1)
            #     model.zero_grad()
            #     ground_truth.grad.data.zero_()
            #     deviation_f1_target[:, f] = 0
            # deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
            # thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), args.pruning_rate)
            # mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

        #***************************Gradient Compression*****************************
            # for i in w:
            #     grad_tensor = w[i].cpu().numpy()
            #     flattened_weights = np.abs(grad_tensor.flatten())
            #     # Generate the pruning threshold according to 'prune by percentage'. (Your code: 1 Line)
            #     thresh = np.percentile(flattened_weights, 70)
            #     grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
            #     w[i] = torch.Tensor(grad_tensor).to(args.device)


        #***********************Fed-NA**************************
            # flag = 0
            # for i in w:
            #     if (i == 'layer_hidden1.weight'):
            #
            #         # The addition of zero-sum noise is determined by the shape of the required model.
            #         w[i] = w[i] + noise_add[target]
            #         flag +=1
            #         continue
            #     elif (flag < 4):
            #         flag += 1
            #         continue
            #     else:
            #         noise_tensor = torch.from_numpy(np.random.laplace(loc=0, scale=0.1, size=w[i].shape)).to(args.device)
            #         w[i] = w[i] + noise_tensor
            #         flag +=1
            #
            #
            # for local_model in local_models:
            #     var = []
            #     for param_name, param_value in local_model.items():
            #         param_value_cpu = param_value.cpu()
            #         param_variance = torch.var(param_value_cpu)
            #         var.append(param_variance)
            #     var = [min(v, 1) for v in var]
            #
            # Layer_risk[idx] = var
            #
            # target += 1

        # ***********************DPFL**************************

        # for i in w:
        #     noise_tensor = torch.from_numpy(np.random.laplace(loc=0, scale=0.1, size=w[i].shape)).to(args.device)
        #     w[i] = w[i] + noise_tensor

        #***************************Outpost**********************************
            # for local_model in local_models:
            #     var = []
            #     for param_name, param_value in local_model.items():
            #         param_value_cpu = param_value.cpu()
            #         param_variance = torch.var(param_value_cpu)
            #         var.append(param_variance)
            #     var = [min(v, 1) for v in var]
            #
            # fim = []
            # flattened_fim = None
            # for i in w:
            #     squared_grad = w[i].clone().pow(2).mean(0).cpu().numpy()
            #     fim.append(squared_grad)
            #     if flattened_fim is None:
            #         flattened_fim = squared_grad.flatten()
            #     else:
            #         flattened_fim = np.append(flattened_fim, squared_grad.flatten())
            #
            # fim_thresh = np.percentile(flattened_fim, 100 - 40)
            #
            # num = 0
            #
            # for i in w:
            #     # pruning
            #     grad_tensor = w[i].cpu().numpy()
            #     flattened_weights = np.abs(grad_tensor.flatten())
            #     thresh = np.percentile(flattened_weights,80)
            #
            #     grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
            #     # noise
            #     noise_base = torch.normal(0, var[num] * 0.8,w[i].shape)
            #
            #     noise_mask = np.where(fim[num] < fim_thresh, 0, 1)
            #     gauss_noise = noise_base * noise_mask
            #     w[i] = (torch.Tensor(grad_tensor) + gauss_noise).to(dtype=torch.float32).to(args.device)
            #
            #     num+=1


            local_models.clear()
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))



        # update global weights
        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f},Loss: {:.2f}".format(iter, acc_t,loss_t))

       # local_acc_test.append(format(local_accu / local_num))
        acc_test.append(acc_t.item())
        loss_test.append(loss_t)

    duration = (time.time() - start_time)
    print("---train  Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))

    localfilerootpath = './log/file/localaccfile'
    testaccfilerootpath = './log/file/testaccfile'
    lossfilerootpath = './log/file/lossfile'
    # if not os.path.exists(accfilerootpath):
    #     os.makedirs(accfilerootpath)

    local_accfile = open(localfilerootpath + '/local_accfile_fed_{}_{}_{}_localepoch{}_iid{}.dat'.
                         format(args.dataset, args.model, args.epochs,args.local_ep,args.iid), "w")
    accfile = open(testaccfilerootpath + '/accfile_fed_{}_{}_{}_localepoch_{}_iid{}_num100.dat'.
    # accfile = open(testaccfilerootpath + '/accfile_fedprox_{}_{}_{}_localepoch_{}_iid{}_num100.dat'.
                   format(args.dataset, args.model, args.epochs,args.local_ep, args.iid), "w")
    lossfile = open(lossfilerootpath + '/lossfile_fed_{}_{}_epoch{}_localepoch_{}_iid{}_num100.dat'.
    #lossfile = open(lossfilerootpath + '/lossfile_fedprox_{}_{}_epoch{}_localepoch_{}_iid{}_num100.dat'.
                    format(args.dataset, args.model, args.epochs, args.local_ep,args.iid), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    for ls in loss_test:
        sls = str(ls)
        lossfile.write(sls)
        lossfile.write('\n')
    lossfile.close()

    accpltrootpath = './log/plt/accplt'
    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(accpltrootpath + '/fed_{}_{}_{}_C{}_localepoch_{}_iid{}_num100_acc.png'.format(args.dataset, args.model, args.epochs, args.frac,args.local_ep, args.iid))
    #plt.savefig(accpltrootpath + '/fedprox_{}_{}_{}_C{}_localepoch{}_iid{}_num100_acc.png'.format(args.dataset, args.model, args.epochs, args.frac,args.local_ep, args.iid))

    losspltrootpath = './log/plt/lossplt'
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('loss accuracy')
    plt.savefig(losspltrootpath + '/fed_{}_{}_{}_C{}_localepoch_{}_iid{}_num100_loss.png'.format(args.dataset, args.model, args.epochs, args.frac,args.local_ep,args.iid))
    # plt.savefig(losspltrootpath + '/fedprox_{}_{}_{}_C{}_localepoch{}_iid{}_num100_loss.png'.format(args.dataset, args.model, args.epochs, args.frac,args.local_ep,args.iid))


