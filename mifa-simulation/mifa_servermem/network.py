#!/usr/bin/env python
# coding: utf-8
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import math

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

class MLP(nn.Module):
    def __init__(self,input_size=784,output_size=62):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 62)
        # self.linear2 = nn.Linear(128, 64)
        # self.linear3 = nn.Linear(128,output_size)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear1(x))
        x = self.linear1(x)

        return(x)

import torch.nn.functional as func
class LeNetmnist(nn.Module):
    def __init__(self):
        super(LeNetmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # 6*6 from image dimension
        self.fc2 = nn.Linear(256, 62)
        # self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class ShakespeareLstm(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim=128,
                 n_recurrent_layers=1, output_dim=128, default_batch_size=32):
        super(ShakespeareLstm, self).__init__()

        # Word embedding
        embedding_dim = 8
        self.embedding = torch.nn.Embedding(num_classes, embedding_dim)

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of stacked lstm layers
        self.n_recurrent_layers = n_recurrent_layers

        # shape of input/output tensors: (batch_dim, seq_dim, feature_dim)
        self.rnn = torch.nn.GRU(embedding_dim, self.hidden_dim, n_recurrent_layers, batch_first=True)
        self.fc = torch.nn
        (self.hidden_dim, output_dim)


    def forward(self, x):
        x = torch.tensor(x).to(torch.int64)
        # word embedding
        x = self.embedding(x)
        # query RNN
        out, _ = self.rnn(x)
        # out, _ = self.rnn(x, (self.h0.detach(), self.c0.detach()))

        # Index hidden state of last time step; out.size = `batch, seq_len, hidden`
        out = self.fc(out[:, -1, :])
        return out

# Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out

# # ResNet
# class Network(nn.Module):
#     def __init__(self, block=2, layers=2, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 16
#         self.conv = conv3x3(3, 16)
#         self.bn = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 16, layers[0])
#         self.layer2 = self.make_layer(block, 32, layers[1], 2)
#         self.layer3 = self.make_layer(block, 64, layers[2], 2)
#         self.avg_pool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64, num_classes)

#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels))
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out



# class Network(nn.Module):
#   def __init__(self):  
#             super().__init__()  
#             self.conv1=nn.Conv2d(3,20,5,1)  
#             self.conv2=nn.Conv2d(20,50,5,1)  
#             self.fully1=nn.Linear(5*5*50,500)  
#             self.dropout1=nn.Dropout(0.5)   
#             self.fully2=nn.Linear(500,10)  
#   def forward(self,x):  
#       x=func.relu(self.conv1(x))  
#       x=func.max_pool2d(x,2,2)  
#       x=func.relu(self.conv2(x))  
#       x=func.max_pool2d(x,2,2)  
#       x=x.view(-1,5*5*50) #Reshaping the output into desired shape  
#       x=func.relu(self.fully1(x)) #Applying relu activation function to our first fully connected layer  
#       x=self.dropout1(x)  
#       x=self.fully2(x)    #We will not apply activation function here because we are dealing with multiclass dataset  
#       return x     

# #Cnn lenet type1
# class Network(nn.Module): 
#   def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1   = nn.Linear(16*5*5, 120)
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 10)

#   def forward(self, x):
#       out = F.relu(self.conv1(x))
#       out = F.max_pool2d(out, 2)
#       out = F.relu(self.conv2(out))
#       out = F.max_pool2d(out, 2)
#       out = out.view(out.size(0), -1)
#       out = F.relu(self.fc1(out))
#       out = F.relu(self.fc2(out))
#       out = self.fc3(out)
#       return out

#CNN - LeNet
# class Network(nn.Module):  
#     def __init__(self, n_classes=10):
#       super(Network, self).__init__()
      
#       self.feature_extractor = nn.Sequential(            
#           nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
#           nn.Tanh(),
#           nn.AvgPool2d(kernel_size=2),
#           nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
#           nn.Tanh(),
#           nn.AvgPool2d(kernel_size=2),
#           nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
#           nn.Tanh()
#       )

#       self.classifier = nn.Sequential(
#           nn.Linear(in_features=120, out_features=84),
#           nn.Tanh(),
#           nn.Linear(in_features=84, out_features=n_classes),
#       )


#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = torch.flatten(x, 1)
#         logits = self.classifier(x)
#         #probs = F.softmax(logits, dim=1)
#         return logits

# #DNN
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()       
#         self.fc1 = nn.Linear(3072, 1536)
#         self.fc2 = nn.Linear(1536, 768)
#         self.fc3 = nn.Linear(768, 384)
#         self.fc4 = nn.Linear(384, 128)
#         self.fc5 = nn.Linear(128, 10)
#         self.relu=nn.ReLU()
        
     

#     def forward(self, x):
                
#         # Flatten images into vectors
#         out = x.view(x.size(0), -1)
#         # Apply layers & activation functions
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         out = self.relu(out)
#         out = self.fc4(out)
#         out = self.relu(out)
#         out = self.fc5(out)
        
#         return out
    
