#!/usr/bin/env python
# coding: utf-8

"""state_dict() returns a pointer to the model, so you cannot do new state_dict - old state_dict and expect them to have different values
   even dict(state_dict) did not seem to return a new object, just a reference to the same model. I used deepcopy to find gradient
"""

from matplotlib.colors import LinearSegmentedColormap
from numpy.random.mtrand import standard_cauchy
import torch
from torch._C import _nccl_all_reduce
import network
from torch.utils.data import Dataset, DataLoader
#import data
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import lenet
import torchvision
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.utils.data as data_utils
import torch.optim as optim
import copy
import config
import os
import data_cifar10 as data
from tqdm import tqdm

def noniid_partition(dataset, num_users):
    """
    Sample non-I.I.D client data
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 2*num_users, int(len(dataset)/(2*num_users))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

class Server():

    def __init__(self, n_c, total_clients, global_lr, cluster =0):
        
        self.n_c = n_c
        self.global_lr = global_lr
        self.total_c = total_clients
        self.global_model = lenet.LeNet()
        self.clients_alllayer_prevgrad = {}
        self.layers_init = self.global_model.state_dict()
        self.cluster = cluster #1 to implement clustering
        

        #initialize each layer's gradient to 0 for a client
        for layer in self.layers_init:
            self.layers_init[layer] = torch.zeros_like(self.layers_init[layer], dtype=torch.float64)
        
        if self.cluster == 1:
            self.n_clusters = config.K #no of clusters

            #Compute the current cluster center value as a weighted running avg
            self.cluster_center_vals = {}
            for i in range(self.n_clusters):
                self.cluster_center_vals[i] = -1
            

        self.running_av_grads = copy.deepcopy(self.layers_init)
      
        #each client
        for c in range(total_clients):
            self.clients_alllayer_prevgrad[c] = copy.deepcopy(self.layers_init)
        
        self.d_algo = {
            0: "MIFA",
            1: "UMIFA",
            2: "FedAvg"
        }
    
    def params_to_vec(self, vec):
        out = []
        for layer in vec:
            out.append(vec[layer].view(-1))
        return torch.cat(out)
    
    def get_and_update_closest_clust_ind(self,client_id,client_obj):

        #iterate over cluster centers (distance:l2)
        min_dist = float('inf')
        new_clust_cent = -1
        client_update_vect = self.params_to_vec(client_obj[client_id].local_grad_update)
        for clust_key, clust_val in self.cluster_center_vals.items():
            if clust_val == -1:
                self.cluster_center_vals[clust_key] = copy.deepcopy(client_obj[client_id].local_grad_update)
                client_obj[client_id].cluster_center = clust_key
                return clust_key
            cluster_vect = self.params_to_vec(self.cluster_center_vals[clust_key])
            dist = torch.linalg.norm(client_update_vect - cluster_vect)
            if dist < min_dist:
                min_dist = dist
                new_clust_cent = clust_key

        #Update client's cluster center index to current cluster 
        if new_clust_cent != -1:
            client_obj[client_id].cluster_center = new_clust_cent
        else:
            print("Cluster key for client ", client_id, " not updated")
            print(dist)
            print(cluster_vect)
            print("clv",client_update_vect)
            print("actual ",client_obj[client_id].local_grad_update)

        
        #return closest cluster index
        return new_clust_cent

    
    def get_cluster_prevgrad(self,id,client_obj):
        clust_ind =  client_obj[id].cluster_center
        if clust_ind != -1:
            return self.cluster_center_vals[clust_ind]
        else:
            return self.layers_init


    def MIFA_add(self, ids, client_models):

        absent = []
        for c in list(range(self.total_c)):
            if c not in ids:
                absent.append(c)
        step = copy.deepcopy(self.layers_init)
        #sum grads from absent clients
        for client in absent:
            grads = copy.deepcopy(self.clients_alllayer_prevgrad[client])
            for layer in step:
                step[layer] += grads[layer]/self.total_c
                #print(layer, np.linalg.norm(self.clients_alllayer_prevgrad[client][layer]))

        

        #sum grads from present clients
        for client in ids:
            grads = copy.deepcopy(client_models[client].local_grad_update)
            for layer in step:
                step[layer] += grads[layer]/self.total_c
                self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(grads[layer])
                #print(client, layer, np.linalg.norm(grads[layer]))
                
        # norm = 0
        # for k,layer in step.items():
        #     print("\n\nk  ",k,layer)
        #     norm += np.linalg.norm(grads[k])
         

        return step

        
    def MIFA(self, ids, client_models):

        if self.cluster ==0:
            #sum grads from present clients
            for client in ids:
                alllayer_grads = client_models[client].local_grad_update
                for layer in self.running_av_grads:
                    layer_grad = alllayer_grads[layer].cpu()
                    prevgrad = self.clients_alllayer_prevgrad[client][layer].cpu() 
                    self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c

                    #self.running_av_grads[layer] += (layer_grad - self.clients_alllayer_prevgrad[client][layer])/self.total_c
                    self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])
            #print(self.running_av_grads[layer])
            return self.running_av_grads
        
        else:
            #sum grads from present clients
            for client in ids:
                alllayer_grads = client_models[client].local_grad_update
                alllayer_prevgrads = self.get_cluster_prevgrad(client,client_models)
                closest_clustcent = self.get_and_update_closest_clust_ind(client,client_models)
                for layer in self.running_av_grads:
                    layer_grad = alllayer_grads[layer].cpu()
                    prevgrad = alllayer_prevgrads[layer].cpu() 
                    self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c

                    #self.running_av_grads[layer] += (layer_grad - self.clients_alllayer_prevgrad[client][layer])/self.total_c
                    self.cluster_center_vals[closest_clustcent][layer] = (0.2*self.cluster_center_vals[closest_clustcent][layer])\
                        + (0.8*alllayer_grads[layer])
            #print(self.running_av_grads[layer])
            return self.running_av_grads



    def FedAvg(self, ids, client_models):
        
        step = copy.deepcopy(self.layers_init)
        #add grads from present clients to running avg
        for client in ids:
            alllayer_grads = client_models[client].local_grad_update
            for layer in alllayer_grads:
                layer_grad = alllayer_grads[layer].cpu()
                step[layer] += layer_grad/self.n_c
            
        return step 

    def UMIFA(self, ids, client_models):

        if self.cluster ==0:
            umifa_step = copy.deepcopy(self.layers_init)
            running_avg_t = copy.deepcopy(self.running_av_grads)
            #add grads from present clients to running avg
            for client in ids:
                alllayer_grads = client_models[client].local_grad_update
                for layer in self.running_av_grads:
                    layer_grad = alllayer_grads[layer].cpu()
                    prevgrad = self.clients_alllayer_prevgrad[client][layer].cpu()
                    umifa_step[layer] += (running_avg_t[layer] + (layer_grad - prevgrad))/self.n_c
                    self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c                
                    self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])
            return umifa_step
        else:
            umifa_step = copy.deepcopy(self.layers_init)
            running_avg_t = copy.deepcopy(self.running_av_grads)
            #add grads from present clients to running avg
            for client in ids:
                alllayer_grads = client_models[client].local_grad_update
                alllayer_prevgrads = self.get_cluster_prevgrad(client,client_models)
                closest_clustcent = self.get_and_update_closest_clust_ind(client,client_models)
                for layer in self.running_av_grads:
                    layer_grad = alllayer_grads[layer].cpu()
                    prevgrad = alllayer_prevgrads[layer].cpu()
                    umifa_step[layer] = (running_avg_t[layer] + layer_grad - prevgrad)/self.n_c
                    self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c                
                    self.cluster_center_vals[closest_clustcent][layer] = (0.2*self.cluster_center_vals[closest_clustcent][layer])\
                        + (0.8*alllayer_grads[layer])
            return umifa_step


    #ids are participating client IDS
    def aggregate(self, ids, client_models, algo):
        
        # print(self.global_model.state_dict())
        if algo == 0:
            step = self.MIFA(ids, client_models)
        elif algo ==1:
            step = self.UMIFA(ids, client_models)
        elif algo ==2:
            step = self.FedAvg(ids, client_models)
        
        else:
            print("Algo not valid: ", algo)

        global_state_dict = self.global_model.state_dict()

        
        for layer in global_state_dict:
            global_state_dict[layer] = global_state_dict[layer].float()
            global_state_dict[layer] += (step[layer])


        self.global_model.load_state_dict(global_state_dict)           


    def test(self, test_data_loader):
    
        pred =[]
        actuals = []
        l=len(test_data_loader)
        test_model = copy.deepcopy(self.global_model).cuda()

        test_model.eval()
        #Over entire test set
        for i in range(l):
            self.global_model.eval()
            test_X, lab=next(iter(test_data_loader))
            test_X = test_X.cuda()
            #forward
            out=torch.argmax(test_model.forward(test_X), axis = 1)

            pred.extend(out)
            actuals.extend(lab)
        accuracy = torch.count_nonzero(torch.Tensor(pred)== torch.Tensor(actuals))
        return accuracy/len(pred)
    
    def allclients_loss(self, client_id):
    
        pred =[]
        actuals = []
        l=len(test_data_loader)
        test_model = copy.deepcopy(self.global_model).cuda()

        test_model.eval()
        #Over entire test set
        for i in range(l):
            self.global_model.eval()
            test_X, lab=next(iter(test_data_loader))
            test_X = test_X.cuda()
            #forward
            out=torch.argmax(test_model.forward(test_X), axis = 1)
            
            pred.extend(out)
            actuals.extend(lab)
        accuracy = torch.count_nonzero(torch.Tensor(pred)== torch.Tensor(actuals))
        return accuracy/len(pred)
    
def plot_clusters(client_obj_dict, n_c, algo, timestr):
    final_clusters = np.zeros(config.K)
    for c in client_obj_dict:
        clust = client_obj_dict[c].cluster_center
        if clust !=-1:
            final_clusters[clust]+=1
    bins = list(range(config.K))
    plt.title("Distribution of clusters after experiment Algo: "+ config.d_algo[algo])
    plt.bar(bins,final_clusters)#,bins = bins, density = False)
    plt.ylabel('Number of clients')
    plt.xlabel('cluster number')
    if not os.path.exists("./n_c_"+str(n_c)+"/clusthist"):
            #cwd = os.getcwd()
            os.makedirs("./n_c_"+str(n_c)+"/clusthist")
    plt.savefig("./n_c_"+str(n_c)+"/clusthist/"+timestr)
    plt.close()



def schedule(server= None, mode = 'local'):
    #schedule
        # if rnd == 500:
        #     fact = 5
        #     if mode =='global':
        #         server.global_lr /=fact
        #     else:
        #         for c in client_object_dict:
        #             for g in client_object_dict[c].optimizer.param_groups:
        #                 g['lr'] = g['lr']/fact

        if rnd == 700:
            fact = 5
            if mode =='global':
                server.global_lr /=fact
            else:
                for c in client_object_dict:
                    for g in client_object_dict[c].optimizer.param_groups:
                        g['lr'] = g['lr']/fact

        # elif rnd == 400:
        #     fact=10
        #     if mode =='global':
        #         server.global_lr /=fact
        #     else:
        #         for c in client_object_dict:
        #             for g in client_object_dict[c].optimizer.param_groups:
        #                 g['lr'] = g['lr']/fact
        # elif rnd == 350:
        #     server.global_lr /=2
        #     for g in self.optimizer.param_groups:
        #         g['lr'] = g['lr']/2

class Client():

    def __init__(self, id, lr, model):
        self.id = id
        self.lr = lr
        model = model 
        #lenet.LeNet()
        self.model = model.cuda()
        weightDecay = 5e-5
        self.criterion= torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weightDecay, momentum = 0.2)
        self.cluster_center = -1
        self.local_grad_update = {}

    
    def train(self,train_data_loader,tau):
        avg_loss=0
        #print(self.model.state_dict())
        # print("\n\n\n\n")
        self.model.train()
        for i, (train_X, lab) in enumerate(train_data_loader):
            #train_X.view(train_X.shape[0],3,32,32).cuda()
            #train_X = train_X.transpose(1,3).cuda() 
            train_X = train_X.cuda()
            lab = lab.cuda()
            self.optimizer.zero_grad() 
            out=self.model.forward(train_X.float())
            loss=self.criterion(out,lab)
            loss.backward()
            self.optimizer.step()
            avg_loss+=loss.data
             
        loss = avg_loss/len(train_data_loader)     
        return loss
    
    def local_train(self,dtrain_loader,tau, rnd, server):

        oldx = copy.deepcopy(self.model.state_dict())
        loss =0
        #train each client for local epochs
        for epoch in range(config.local_epochs):
            loss += self.train(dtrain_loader,tau)
        newx = copy.deepcopy(self.model.state_dict())
        #print("loss",loss)
        for layer, val in newx.items():
            self.local_grad_update[layer] = (newx[layer]-oldx[layer])

        return loss/config.local_epochs
        
    def setWeights(self, global_model_sd):
        self.model.load_state_dict(global_model_sd)


def plot(global_loss, global_acc,n_c, timestr):
    i=0
    for loss_per_lr in global_loss:
        plt.figure(figsize=(10,7)) 
        plt.ylabel('Loss')
        plt.title('Global Loss vs Comm rounds Learning Rate = {0}'.format(possible_lr[i]))  
        plt.xlabel('Number of Comm rounds')
        for algo_i, loss_per_algo in enumerate(loss_per_lr):
            plt.plot(np.arange(len(loss_per_algo))*config.plot_every_n, loss_per_algo, label = d_algo[d_algo_keys[algo_i]])
        plt.legend(loc = 'upper right')
        dir_name ="./n_c_"+str(n_c)+"/train/{0}.png".format(timestr)
        if not os.path.exists("./n_c_"+str(n_c)+"/train"):
            #cwd = os.getcwd()
            os.makedirs("./n_c_"+str(n_c)+"/train")
        plt.savefig(dir_name)
        i+=1
    i=0
    for acc_per_lr in global_acc:
        plt.figure(figsize=(10,7)) 
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy vs Comm rounds Learning Rate = {0}'.format(possible_lr[i]))  
        plt.xlabel('Number of Comm rounds')
        for algo_i, acc_per_algo in enumerate(acc_per_lr):
            plt.plot(np.arange(len(acc_per_algo))*config.plot_every_n, acc_per_algo, label = d_algo[d_algo_keys[algo_i]])
        plt.legend(loc = 'lower right')
        dir_name ="./n_c_"+str(n_c)+"/test/{0}.png".format(timestr)
        if not os.path.exists("./n_c_"+str(n_c)+"/test"):
            #cwd = os.getcwd()
            os.makedirs("./n_c_"+str(n_c)+"/test")
        plt.savefig(dir_name)
        i+=1
    #plt.close()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    #Init hyperparams
    client_names=list(range(config.total_c))
    batch_size=config.batch_size
    n_rnds=config.n_rnds
    tau=config.tau
    total_c=config.total_c
    possible_lr=config.possible_lr
    #global_lr = config.global_lr

    
    #Init data
    # train_data,test_data,p_i = data.init_dataset()

    # for c in range(0,100,20):
    #     dtrain_loader=data.get_train_data_loader(train_data, c,batch_size=1)
    #     for images, _ in dtrain_loader:
    #         print(images.shape)
    #         im = torch.reshape(images, (1,3,32,32))
    #         print('images.shape:', images.shape)
    #         #plt.figure(figsize=(25,20))
    #         plt.axis('off')
    #         grid = torchvision.utils.make_grid(im, nrow=1).permute((1,2,0))
    #         print(grid.shape)
    #         plt.imshow(grid)
    #         plt.show()
    #         plt.imshow(images[0])
    #         plt.show()
    #         #plt.savefig('client_dataimages/'+str(c)+'.png')
    #         #plt.imsave('client_dataimages/'+str(c)+'.png',torchvision.utils.make_grid(images, nrow=20).permute((1,2,0)))
            
    #         break

        
    # dtest_loader=data.get_test_data_loader(test_data, batch_size= batch_size)

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
    dict_users = noniid_partition(dataset_train, config.total_c)
    d_algo = config.d_algo
    d_algo_keys = sorted(list(d_algo.keys()))

    client_data_split_dict = {}

    for c in client_names:
        client_data_split_dict[c] = DatasetSplit(dataset_train,dict_users[c])

    # 0 corresponds to vanilla reg_mifa
    # 1 corresponds to umifa
    # 2 corresponds to fedavg
    cmodel = lenet.LeNet()
    timestr = time.strftime("%Y%m%d-%H%M%S")

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

                if algo == 0:
                   global_lr = lr = 0.2
                else:
                    global_lr = lr = 0.1 

                torch.cuda.empty_cache()
                print("--------------------------Algo: {0}------------------------------------".format(d_algo[algo]))     

                server = Server(choose_nc, total_c, global_lr, cluster = config.cluster)
                client_object_dict = {}

                for c in client_names:
                    client_object_dict[c]= Client(c, lr, copy.deepcopy(cmodel))
                
                local_rnd_loss=[]
                local_rnd_acc=[]

                #Global epochs
                for rnd in tqdm(range(config.n_rnds), total = config.n_rnds): # each communication round
                    #schedule(server, mode = 'global')
                    #schedule()

                    idxs_users=idxs_users_allrounds[rnd]
                    # print("chosen clients",idxs_users)
                    idxs_len=len(idxs_users)

                    #Obtain global weights
                    global_state_dict=server.global_model.state_dict()
                
                    #Assign each client model to global model
                    for c in idxs_users:  
                        client_object_dict[c].setWeights(global_state_dict)


                    d_train_losses=0 #loss over all clients
                    for sel_idx in idxs_users:
                        
                        #Get train loader
                        dtrain_loader = DataLoader(client_data_split_dict[sel_idx], batch_size=config.batch_size, shuffle=True)

                        loss = client_object_dict[sel_idx].local_train(dtrain_loader, tau, rnd, server)
                        # print(loss)


                        d_train_losses+=loss
                    d_train_losses /= idxs_len


                    #aggregate at server
                    server.aggregate(idxs_users, client_object_dict, algo)
                    if rnd%config.plot_every_n == 0: 
                        dtest_loader = DataLoader(dataset_test,batch_size = 128, shuffle = False)
                        acc = server.test(dtest_loader)
                        local_rnd_acc.append(acc)
                        local_rnd_loss.append(d_train_losses)
                        print("Testing acc: ",acc)
                    # if(rnd==100):
                    #     lr = lr/2


                    print("Rnd {4} - Training Error: {0}, Algo: {2}, n_c: {3}, lr: {1}\n".format(d_train_losses,lr, algo, choose_nc, rnd))
                loss_algo.append(local_rnd_loss)  
                acc_algo.append(local_rnd_acc) 
                plot_clusters(client_object_dict, choose_nc,algo, timestr)
            loss_eachlr.append(loss_algo)
            acc_eachlr.append(acc_algo)
        
        plot(loss_eachlr, acc_eachlr, choose_nc, timestr)            
        

                
                
                    

