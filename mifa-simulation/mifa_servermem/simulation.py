#!/usr/bin/env python
# coding: utf-8

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

class Server():

    def __init__(self, n_c, total_clients, global_lr):
        
        self.n_c = n_c
        self.global_lr = global_lr
        self.total_c = total_clients
        self.global_model = network.Network()
        self.clients_alllayer_prevgrad = {}
        self.layers_init = self.global_model.state_dict()

        #initialize each layer's gradient to 0 for a client
        for layer in self.layers_init:
            self.layers_init[layer] = torch.zeros_like(self.layers_init[layer])

        self.running_av_grads = copy.deepcopy(self.layers_init)

        #each client
        for c in range(total_clients):
            self.clients_alllayer_prevgrad[c] = copy.deepcopy(self.layers_init)
        
        self.d_algo = {
            0: "MIFA",
            1: "UMIFA",
            2: "FedAvg"
        }
        
    def MIFA(self, ids, client_models):

        #sum grads from present clients
        for client in ids:
            alllayer_grads = client_models[client].local_grad_update
            for layer in self.running_av_grads:
                self.running_av_grads[layer] += (alllayer_grads[layer] - self.clients_alllayer_prevgrad[client][layer])/self.total_c
                self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])
        
        return self.running_av_grads


    def FedAvg(self, ids, client_models):
        
        step = copy.deepcopy(self.layers_init)
        #add grads from present clients to running avg
        for client in ids:
            alllayer_grads = client_models[client].local_grad_update
            for layer in alllayer_grads:
                step[layer] += alllayer_grads[layer]/self.n_c
            
        return step 

    def UMIFA(self, ids, client_models):
        
        umifa_step = copy.deepcopy(self.layers_init)
        #add grads from present clients to running avg
        for client in ids:
            alllayer_grads = client_models[client].local_grad_update
            for layer in self.running_av_grads:
                umifa_step[layer] = self.running_av_grads[layer] + (alllayer_grads[layer] - self.clients_alllayer_prevgrad[client][layer])/self.n_c
                self.running_av_grads[layer] += (alllayer_grads[layer] - self.clients_alllayer_prevgrad[client][layer])/self.total_c                
                self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])
        return umifa_step

    #ids are participating client IDS
    def aggregate(self, ids, client_models, algo):
        
        # print(self.global_model.state_dict())
        print(ids)
        if algo == 0:
            step = self.MIFA(ids, client_models)
        elif algo == 1:
            step = self.UMIFA(ids, client_models)
        elif algo ==2:
            step = self.FedAvg(ids, client_models)
        else:
            print("Algo not valid: ", algo)

        global_state_dict = self.global_model.state_dict()

        for layer in global_state_dict:
            global_state_dict[layer] -= (self.global_lr * step[layer])

        self.global_model.load_state_dict(global_state_dict)           


    def test(self, test_data_loader):
    
        pred =[]
        actuals = []
        l=len(test_data_loader)

        #Local epochs
        for i in range(l):
            test_X, lab=next(iter(test_data_loader))
                    
            #forward
            out=torch.argmax(self.global_model.forward(test_X), axis = 1)
            pred.extend(out)
            actuals.extend(lab)
        accuracy = torch.count_nonzero(torch.Tensor(pred)== torch.Tensor(actuals))
        return accuracy/len(pred)




class Client():

    def __init__(self, id, lr):
        self.id = id
        self.lr = lr
        self.model = network.Network()
        self.criterion= torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.local_grad_update = {}

    
    def train(self,train_data_loader,tau):
        avg_loss=0
        #print(self.model.state_dict())
        # print("\n\n\n\n")
        for i, (train_X, lab) in enumerate(train_data_loader):
        #Local epochs
            self.optimizer.zero_grad() 
            out=self.model.forward(train_X.float())
            loss=self.criterion(out,lab)
            # print(loss.data)
            #print("before",self.model.l1.weight.grad, loss)

            loss.backward()
            #print("after",self.model.l1.weight.grad)
            self.optimizer.step()
            avg_loss+=loss.data
            #print(loss.data)
            if i==tau:
                break
       
        return avg_loss/tau
    
    def local_train(self,dtrain_loader,tau):

        oldx = copy.deepcopy(self.model.state_dict())
        loss = self.train(dtrain_loader,tau)
        newx = copy.deepcopy(self.model.state_dict())

        for layer in newx:
            self.local_grad_update[layer] = (oldx[layer] - newx[layer])/self.lr

        return loss
        
    def setWeights(self, global_model_sd):
        self.model.load_state_dict(global_model_sd)


    
    


def plot(global_loss, global_acc,n_c):
    i=0
    print(len(global_loss),len(global_loss[0]))
    for loss_per_lr in global_loss:
        plt.figure(figsize=(10,7)) 
        plt.ylabel('Loss')
        plt.title('Global Loss vs Comm rounds Learning Rate = {0}'.format(possible_lr[i]))  
        plt.xlabel('Number of Comm rounds')
        for algo_i, loss_per_algo in enumerate(loss_per_lr):
            plt.plot(np.arange(len(loss_per_algo)), loss_per_algo, label = d_algo[d_algo_keys[algo_i]])
        plt.legend(loc = 'lower right')
        dir_name ="n_c_"+str(n_c)+"/train/{0}.png".format(possible_lr[i])
        if not os.path.exists("n_c_"+str(n_c)+"/train"):
            cwd = os.getcwd()
            os.makedirs(cwd+"/n_c_"+str(n_c)+"/train")
        plt.savefig(dir_name)
        i+=1
    i=0
    for acc_per_lr in global_acc:
        plt.figure(figsize=(10,7)) 
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Comm rounds Learning Rate = {0}'.format(possible_lr[i]))  
        plt.xlabel('Number of Comm rounds')
        for algo_i, acc_per_algo in enumerate(acc_per_lr):
            plt.plot(np.arange(len(acc_per_algo)), acc_per_algo, label = d_algo[d_algo_keys[algo_i]])
        plt.legend(loc = 'lower right')
        dir_name ="n_c_"+str(n_c)+"/test/{0}.png".format(possible_lr[i])
        if not os.path.exists("n_c_"+str(n_c)+"/test"):
            cwd = os.getcwd()
            os.makedirs(cwd+"/n_c_"+str(n_c)+"/test")
        plt.savefig(dir_name)
        i+=1
    #plt.close()


if __name__ == "__main__":

    #Init hyperparams
    client_names=list(range(config.total_c))
    batch_size=config.batch_size
    n_rnds=config.n_rnds
    tau=config.tau
    total_c=config.total_c
    possible_lr=config.possible_lr
    #global_lr = config.global_lr

    
    #Init data
    train_data,test_data,p_i = data.init_dataset()
    dtest_loader=data.get_test_data_loader(test_data, batch_size= batch_size)
    d_algo = {
        0: "MIFA",
        1: "UMIFA",
        2: "FedAvg"
    }
    d_algo_keys = sorted(list(d_algo.keys()))

    # 0 corresponds to vanilla reg_mifa
    # 1 corresponds to umifa
    # 2 corresponds to fedavg


    for choose_nc in config.no_of_c:

        loss_eachlr=[] 
        acc_eachlr=[]

        for learning_rate in possible_lr:

            global_lr = learning_rate
            lr=learning_rate
            loss_algo=[] 
            acc_algo =[] 
            idxs_users_allrounds = [np.random.choice(client_names,choose_nc,replace = False) for i in range(config.n_rnds)]


            #Run simulation for each algorithm
            for algo in list(d_algo.keys()): 

                print("--------------------------Algo: {0}------------------------------------".format(d_algo[algo]))     

                server = Server(choose_nc, total_c, global_lr)
                client_object_dict = {}

                for c in client_names:
                    client_object_dict[c]= Client(c, learning_rate)
                
                local_rnd_loss=[]
                local_rnd_acc=[]

                #Global epochs
                for rnd in range(config.n_rnds): # each communication round

                    idxs_users=idxs_users_allrounds[rnd]
                    print("chosen clients",idxs_users)
                    idxs_len=len(idxs_users)

                    #Obtain global weights
                    global_state_dict=server.global_model.state_dict()
                
                    #Assign each client model to global model
                    for c in idxs_users:  
                        client_object_dict[c].setWeights(global_state_dict)


                    d_train_losses=0 #loss over all clients
                    for sel_idx in idxs_users:
                        
                        #Get train loader
                        dtrain_loader=data.get_train_data_loader(train_data, sel_idx,batch_size)
                        loss = client_object_dict[sel_idx].local_train(dtrain_loader, tau)
                        # print(loss)


                        d_train_losses+=loss
                    d_train_losses /= idxs_len


                    #aggregate at server
                    server.aggregate(idxs_users, client_object_dict, algo)
                    
                    acc = server.test(dtest_loader)
                    local_rnd_acc.append(acc)
                    # if(rnd==100):
                    #     lr = lr/2

                    print("\nRnd {5} - Training Error: {0}, Testing Accuracy: {1}, Algo: {2}, n_c: {3}, local lr: {4} ".format(d_train_losses,acc, algo, choose_nc, learning_rate, rnd))
                loss_algo.append(local_rnd_loss)  
                acc_algo.append(local_rnd_acc) 
            loss_eachlr.append(loss_algo)
            acc_eachlr.append(acc_algo)
        
        plot(loss_eachlr, acc_eachlr, choose_nc)            
        

                
                
                    

