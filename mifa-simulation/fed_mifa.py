#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""state_dict() returns a pointer to the model, so you cannot do new state_dict - old state_dict and expect them to have different values
   even dict(state_dict) did not seem to return a new object, just a reference to the same model. I used deepcopy to find gradient
"""

from matplotlib.colors import LinearSegmentedColormap
from numpy.random.mtrand import standard_cauchy
import torch
import network
import data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import torch.utils.data as data_utils
import torch.optim as optim
import copy
import config
import os



# In[3]
#forward prop
#backprop

#test
def test(model,test_data_loader):
    
    acc=0
    samples=0
    l=len(test_data_loader)
    #Local epochs
    for i in range(l):
        test_X, lab=next(iter(test_data_loader))
                
        #forward
        out=model.forward(test_X)
        for i in range(len(lab)):
            if(torch.argmax(out[i])==lab[i]):
                acc+=1
                samples+=0

    return acc/samples


#Regular MIFA
def reg_mifa(sel_idx,learning_rate,criterion, dtrain_loader, tau,client_state_dict):
    #copy model weights
    temp_state_dict = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict()))

    #Train for tau local epochs
    loss=train(client_model_dict[sel_idx],learning_rate,criterion, dtrain_loader, tau)

    client_state_dict[sel_idx] = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict())) #trained model weights
    client_prev_state_dict = client_prev_model_dict[sel_idx].state_dict() #holds the stale gradient of client
    for key in client_state_dict[sel_idx]:
        prev_weights = temp_state_dict[key].type(torch.DoubleTensor)
        new_weights =client_state_dict[sel_idx][key].type(torch.DoubleTensor) #gradient after tau local epochs
        current_grad = (prev_weights/lr) - (new_weights/lr)
        update = (current_grad/p_i[sel_idx]) - ((1-(1/p_i[sel_idx]))*client_prev_state_dict[key])  # gradient - prev gradient
        client_prev_state_dict[key] = client_state_dict[sel_idx][key]= update #set prev gradient to current gradient
    client_prev_model_dict[sel_idx].load_state_dict(client_prev_state_dict)
    return loss,client_state_dict

#saga Implemented for gradients sent by clients to server, saga during aggregation
def saga_agg(sel_idx,learning_rate,criterion, dtrain_loader, tau,client_state_dict):
    #copy model weights
    temp_state_dict = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict()))

    #Train for tau local epochs
    loss=train(client_model_dict[sel_idx],learning_rate,criterion, dtrain_loader, tau)

    client_state_dict[sel_idx] = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict())) #trained model weights
    for key in client_state_dict[sel_idx]:
        prev_weights = temp_state_dict[key].type(torch.DoubleTensor)
        new_weights =client_state_dict[sel_idx][key].type(torch.DoubleTensor) #gradient after tau local epochs
        current_grad = (prev_weights/lr) - (new_weights/lr)
        client_state_dict[sel_idx][key] = current_grad  # gradient 
        #client_prev_state_dict[key] = current_grad #set prev gradient to current gradient
    #client_prev_model_dict[sel_idx].load_state_dict(client_prev_state_dict)
    return loss,client_state_dict

#saga
def saga(sel_idx,learning_rate,criterion, dtrain_loader, tau,client_state_dict):
    #copy model weights
    temp_state_dict = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict()))

    #Train for tau local epochs
    loss=train(client_model_dict[sel_idx],learning_rate,criterion, dtrain_loader, tau)

    client_state_dict[sel_idx] = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict())) #trained model weights
    client_prev_state_dict = client_prev_model_dict[sel_idx].state_dict() #holds the stale gradient of client
    for key in client_state_dict[sel_idx]:
        prev_weights = temp_state_dict[key].type(torch.DoubleTensor)
        new_weights =client_state_dict[sel_idx][key].type(torch.DoubleTensor) #gradient after tau local epochs
        current_grad = (prev_weights/lr) - (new_weights/lr)
        client_state_dict[sel_idx][key] = current_grad - client_prev_state_dict[key]  # gradient - prev gradient
        client_prev_state_dict[key] = current_grad #set prev gradient to current gradient
    client_prev_model_dict[sel_idx].load_state_dict(client_prev_state_dict)
    return loss,client_state_dict


#Unbiased MIFA
def unbiased_mifa(sel_idx,learning_rate,criterion, dtrain_loader, tau,client_state_dict):
    #copy model weights
    temp_state_dict = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict()))

    #Train for tau local epochs
    loss=train(client_model_dict[sel_idx],learning_rate,criterion, dtrain_loader, tau)

    client_state_dict[sel_idx] = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict())) #trained model weights
    client_prev_state_dict = client_prev_model_dict[sel_idx].state_dict() #holds the stale gradient of client
    for key in client_state_dict[sel_idx]:
        prev_weights = temp_state_dict[key].type(torch.DoubleTensor)
        new_weights =client_state_dict[sel_idx][key].type(torch.DoubleTensor) #gradient after tau local epochs
        current_grad = (prev_weights/lr) - (new_weights/lr)
        client_state_dict[sel_idx][key] = (current_grad - client_prev_state_dict[key])  # gradient - prev gradient
        client_prev_state_dict[key] = current_grad #set prev gradient to current gradient
    client_prev_model_dict[sel_idx].load_state_dict(client_prev_state_dict)
    return loss,client_state_dict

#FedAvg
def fedavg(sel_idx,learning_rate,criterion, dtrain_loader, tau,client_state_dict):
    #copy model weights
    temp_state_dict = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict()))

    #Train for tau local epochs
    loss=train(client_model_dict[sel_idx],learning_rate,criterion, dtrain_loader, tau)

    client_state_dict[sel_idx] = copy.deepcopy(dict(client_model_dict[sel_idx].state_dict())) #trained model weights
    for key in client_state_dict[sel_idx]:
        prev_weights = temp_state_dict[key].type(torch.DoubleTensor)
        new_weights =client_state_dict[sel_idx][key].type(torch.DoubleTensor) #gradient after tau local epochs
        client_state_dict[sel_idx][key]  = (prev_weights/lr) - (new_weights/lr)
    return loss,client_state_dict
    
    


# In[6]:


#1 comm round for each client
def train(model,lr,criterion,train_data_loader,tau):
    avg_loss=0
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_data = iter(train_data_loader)
    #Local epochs
    for i in range(tau):
        train_X, lab=next(train_data)
        optimizer.zero_grad() 
        out=model.forward(train_X.float())
        loss=criterion(out,lab)
        loss.backward()
        optimizer.step()
        avg_loss+=loss.data

    return avg_loss/tau
        
    


# In[7]:


def plot(global_loss,n_c):
    i=0
    print(len(global_loss),len(global_loss[0]))
    for loss_per_lr in global_loss:
        plt.figure(figsize=(10,7)) 
        plt.ylabel('Loss')
        plt.title('Global Loss vs Comm rounds Learning Rate = {0}'.format(possible_lr[i]))  
        plt.xlabel('Number of Comm rounds')
        for algo_i, loss_per_algo in enumerate(loss_per_lr):
            plt.plot(np.arange(len(loss_per_algo)), loss_per_algo, label = d_algo[algo_i])
        plt.legend(loc = 'lower right')
        plt.ylim(0,2)
        dir_name ="n_c_"+str(n_c)+"/{0}.png".format(possible_lr[i])
        if not os.path.exists("n_c_"+str(n_c)):
            os.mkdir("n_c_"+str(n_c))
        plt.savefig(dir_name)
        i+=1


# In[10]:

client_names=list(range(100))

p_c=np.array(client_names)/len(client_names)
batch_size=config.batch_size
n_rnds=config.n_rnds
tau=config.tau
local_m=config.local_m
no_of_c=[10,20,30,40,50,60]#no of clients to choose in each round


train_data,test_data,p_i = data.init_dataset()
stalegrads_state_dict={}
possible_lr=[1,0.5,0.1, 0.05,0.025] 
d_algo = {
    0: "MIFA",
    1: "UMIFA",
    2: "SAGA",
    3: "FedAvg"
}

# 0 corresponds to vanilla reg_mifa
# 1 corresponds to umifa
# 2 corresponds to sarah
# 3 corresponds to fedavg
 
for n_c in no_of_c:
    loss_eachlr=[] 
    for learning_rate in possible_lr:
        lr=learning_rate
        loss_algo=[]  
        gamma = learning_rate/3
        for algo in list(d_algo.keys()):      

            model = network.Network() #global model holding weights wt
            criterion= torch.nn.CrossEntropyLoss()
            global_average_state_dict=model.state_dict()
            #Dictionary that stores the models and optimizers of each client
            client_model_dict={}
            client_prev_model_dict={}

            #holds grad-prev_grad 
            stalegrads_state_dict={}

            for c in client_names:
                client_model_dict[c]=network.Network()
                client_prev_model_dict[c]=network.Network()
                
                #Set initial prev gradients to 0
                client_prev_state_dict = client_prev_model_dict[c].state_dict()
                for key in client_prev_state_dict:
                    client_prev_state_dict[key]=torch.zeros(client_prev_state_dict[key].size())

                client_prev_model_dict[c].load_state_dict(client_prev_state_dict)
                
            

            #initialize G_bar_t to 0
            for key in global_average_state_dict:
                global_average_state_dict[key] = torch.zeros(global_average_state_dict[key].size())
            
            local_rnd_loss=[]

            #Global epochs
            for rnd in range(n_rnds): # each communication round
                if(rnd==0):
                    idxs_users = client_names
                elif config.choose_nc == True:
                    idxs_users = np.random.choice(client_names,n_c,replace = False,p=p_i/sum(p_i))
                else:
                    idxs_users=[]
                    for c in client_names:
                        if np.random.choice([True, False],p=[p_i[c],1-p_i[c]]) ==True: 
                            idxs_users.append(c)
                print("chosen clients",idxs_users)
                
                #Obtain global weights
                global_state_dict=model.state_dict()
            
                #Assign each client model to global model
                for c in idxs_users:            
                    client_model_dict[c].load_state_dict(global_state_dict)
                d_train_losses=0 #per client loss

                #iterate through selected clients
                for sel_idx in idxs_users:

                    #Get train loader
                    dtrain_loader=data.get_train_data_loader(train_data, sel_idx,batch_size)

                    if algo ==0:
                        loss, stalegrads_state_dict= reg_mifa(sel_idx,lr,criterion, dtrain_loader, tau, stalegrads_state_dict)
                    elif algo ==1: 
                        loss, stalegrads_state_dict= unbiased_mifa(sel_idx,lr,criterion, dtrain_loader, tau, stalegrads_state_dict)
                    elif algo ==2:
                        loss, stalegrads_state_dict= saga_agg(sel_idx,lr,criterion, dtrain_loader, tau, stalegrads_state_dict) #we want to return stale 
                    elif algo == 3:
                        loss, stalegrads_state_dict= fedavg(sel_idx,lr,criterion, dtrain_loader, tau, stalegrads_state_dict)
                    else:
                        print("Wrong algo chosen, Abort")
                        exit(1)
                    
                    d_train_losses+=loss
                d_train_losses = d_train_losses/n_c

                print("Round_%d Loss:%f Algo:%d\n\n"%(rnd,d_train_losses,algo))


                local_rnd_loss.append(d_train_losses)
                

                denom = local_m
                if(algo==3):
                    denom = n_c
                    for key in global_average_state_dict:
                        global_average_state_dict[key] = torch.zeros(global_average_state_dict[key].size())
                
                # global_average_state_dict holds the average gradient of all n clients (including stale clients), 
                # except in saga, where it holds 1/n(sum of (current grad_i + prev_grad_i))  
                for clients in idxs_users:                    
                    clienti_local_state_dict=stalegrads_state_dict[clients]
                    for key in global_average_state_dict:      
                        #print(p_i[clients])                  
                        global_average_state_dict[key] += (clienti_local_state_dict[key]/denom)
                
                saga_global_av_sd = copy.deepcopy(global_average_state_dict) 

                global_lr = learning_rate
                if (algo==2):
                    global_lr = gamma
                    for clients in idxs_users:
                        clienti_local_state_dict =copy.deepcopy(stalegrads_state_dict[clients])
                        client_prev_state_dict = copy.deepcopy(client_prev_model_dict[clients].state_dict()) #previous gradient of client                      
                        for key in clienti_local_state_dict:                        
                            saga_update = saga_global_av_sd[key] + clienti_local_state_dict[key] -(client_prev_state_dict[key] * ((local_m+1)/local_m)) #update weights from averaged grad
                            global_state_dict[key] -= (global_lr*saga_update)/n_c
                        client_prev_model_dict[clients].load_state_dict(clienti_local_state_dict)
                #At this point, saga_global_av_sd has  (curr grad - prev_grad + average of all grads /n )     
                #Only update is remaining
                else:                   
                    for key in global_state_dict:
                        global_state_dict[key] -= (global_lr*saga_global_av_sd[key]) #update weights from averaged grad
            
                model.load_state_dict(global_state_dict)
            loss_algo.append(local_rnd_loss)  
        loss_eachlr.append(loss_algo)
                
    plot(loss_eachlr,n_c)

            
            
                

