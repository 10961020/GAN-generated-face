#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/28 20:48
# @Author  : zt
import torch.nn as nn
import torch


class fake_spotter(nn.Module):
    def __init__(self):
        super(fake_spotter, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(54, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Linear(512, 1))

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
