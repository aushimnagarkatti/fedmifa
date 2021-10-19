#!/usr/bin/env python
# coding: utf-8
#
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=2)
        self.conv3 = nn.Conv2d(16, 22, kernel_size=3, stride =2)
        #self.conv4 = nn.Conv2d(22, 16, kernel_size=3, stride =2)
        self.fc1 = nn.Linear(4312, 400)
        self.fc2 = nn.Linear(400, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bn = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(22)


    def forward(self, x):
        x = func.relu(self.bn(self.conv1(x)))
        # x = func.max_pool2d(x, 2)
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.relu(self.bn3(self.conv3(x)))
        # x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x