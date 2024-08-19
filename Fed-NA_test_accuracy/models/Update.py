#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, w_glob,net):
        net.load_state_dict(w_glob)

        net.train()
        #global_parameters = self.weight.values()  # 全局的 weight

        global_parameters = []

        global_weight_collector = list(net.cuda().parameters())
        global_parameters = w_glob
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)#调整学习率

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # print(list(log_probs.size()))
                # print(labels)
                proximal_term = 0.0
                mu = self.args.mu
                # iterate through the current and global model parameters
                # 将当前的 model转化到 cpu 上 计算当前 值

                for param_index, param in enumerate(net.parameters()):
                    proximal_term += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)

                # for w, w_t in zip(net.parameters(), global_parameters):
                #     # update the proximal term
                #     # proximal_term += torch.sum(torch.abs((w-w_t)**2))
                #     proximal_term += (w - w_t).norm(2)  # 需要 将 当前 权重参数转化 为 cpu 才可以进行运算

                # cos = torch.nn.CosineSimilarity(dim=-1)
                # cos(net.parameters(),global_weight_collector)

                loss = self.loss_func(log_probs, labels) + (mu / 2) * proximal_term
                #loss = self.loss_func(log_probs, labels)
                loss.backward()

                # 对参数进行一些操作
                # for param in  net.parameters():
                #     dist.all_reduce(param.grad.data)

                optimizer.step()
                scheduler.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0]

