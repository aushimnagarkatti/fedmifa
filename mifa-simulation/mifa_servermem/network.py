#!/usr/bin/env python
# coding: utf-8
#
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()       
        self.l1=nn.Linear(784,10)
        # self.l2=nn.ReLU()  
        # self.l3 = nn.Linear(128,10)
        
     

    def forward(self, x):
                
        return self.l1(x) #self.l3(self.l2(self.l1(x)))
    
