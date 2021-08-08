#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from data import get_data_loader
from network import Network


# In[2]:

def get_lr(optimizer):
    #TODO: Returns the current Learning Rate being used by
    # the optimizer
    return optimizer.param_groups[0]['lr']


def run(net, epoch, loader, optimizer, criterion, logger, scheduler, train=True):
    # TODO: Performs a pass over data in the provided loader
    
   
    # TODO: Initalize the different Avg Meters for tracking loss and accuracy (if test)

    # TODO: Iterate over the loader and find the loss. Calculate the loss and based on which
    # set is being provided update you model. Also keep track of the accuracy if we are running
    # on the test set
    avg_loss=0
    for i,(images,labels) in enumerate(loader):
        inc=0
        optimizer.zero_grad()
        output=net.forward(images)
        loss=criterion(output,labels)
        
        #Update accuracy
        for d in range(len(labels)):
            if(torch.argmax(output[d])==labels[d]):
                inc+=1
        if(train==True):            
            loss.backward()
            optimizer.step()
        
        avg_loss+=loss
        
        
    avg_loss=avg_loss/
    acc=
    # TODO: Log the training/testing loss using tensorboard. 
#     if(train==True):
#         writer.add_scalar('Loss/train', avg_loss, i)
#     else:
#         writer.add_scalar('Accuracy/test', acc, i)
    # TODO: return the average loss, and the accuracy (if test set)
    return avg_loss,acc
        

def train(net, train_loader, test_loader, logger,tau):    
    # TODO: Define the SGD optimizer here. Use hyper-parameters from cfg
    optimizer = optim.SGD(net.parameters(), lr=0.1/tau)
    # TODO: Define the criterion (Objective Function) that you will be using
    criterion = nn.NLLLoss()
    # TODO: Define the ReduceLROnPlateau scheduler for annealing the learning rate
    weight_visn1=torch.tensor([])
    weight_visn2=torch.tensor([])
    for i in range(2):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, acc = run(net, i, train_loader, optimizer, criterion, logger, scheduler)
        writer.add_scalar('Loss/train', loss, i)
        # TODO: Get the current learning rate by calling get_lr() and log it to tensorboard
        cur_lr=get_lr(optimizer)
        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])
            
        
        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # TODO: HINT - you might need to perform some step before and after running the network
            # on the test set: don't backpropagate on testset
            # Run the network on the test set, and get the loss and accuracy on the test set 
            loss, acc = run(net, i, test_loader, optimizer, criterion, logger, scheduler, train=False)
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc*100.0)
            log_print(log_text, color='red', attrs=['bold'])

            # TODO: Perform a step on the scheduler, while using the Accuracy on the test set
            scheduler.step(acc)
            # TODO: Use tensorboard to log the Test Accuracy and also to perform visualization of the 
            # 2 weights of the first layer of the network!
            writer.add_scalar('Accuracy/test', acc, i)
            writer.add_scalar('Loss/test', loss, i)
            wt1=net.l1.weight[0].reshape(1,1,28,28)
            wt2=net.l1.weight[1].reshape(1,1,28,28)
            weight_visn1=torch.cat( (weight_visn1,wt1),dim=0)
            weight_visn2=torch.cat( (weight_visn2,wt2),dim=0)
            writer.add_scalar('lr', cur_lr,i)
    writer.add_image('Node1',  weight_visn1, 0,dataformats='NCHW')
    writer.add_image('Node2',  weight_visn2, 0,dataformats='NCHW')
                
            
            


# In[3]:


if __name__ == '__main__':
    # TODO: Create a network object
    net = Network()

    # TODO: Create a tensorboard object for logging
    writer = SummaryWriter()
#     grid = torchvision.utils.make_grid(images)
#     writer.add_image('images', grid, 0)
#     writer.add_graph(net, images)
    

    # TODO: Create train data loader
    train_loader = get_data_loader('train')

    # TODO: Create test data loader
    test_loader = get_data_loader('test')

    # Run the training!
    train(net, train_loader, test_loader, writer)







