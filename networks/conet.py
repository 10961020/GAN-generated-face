#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/29 15:01
# @Author  : zt
import torch.nn as nn


class conet(nn.Module):
    def __init__(self):
        super(conet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.fc = nn.Sequential(nn.Linear(4608, 256), nn.Linear(256, 256), nn.Linear(256, 1))

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class cross_conet(nn.Module):
    def __init__(self):
        super(cross_conet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5), nn.MaxPool2d(kernel_size=3))
        self.fc = nn.Sequential(nn.Linear(4608, 256), nn.Linear(256, 256), nn.Linear(256, 1))

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
