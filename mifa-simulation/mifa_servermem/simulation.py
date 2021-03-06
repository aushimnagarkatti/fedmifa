#!/usr/bin/env python
# coding: utf-8

"""state_dict() returns a pointer to the model, so you cannot do new state_dict - old state_dict and expect them to have different values
   even dict(state_dict) did not seem to return a new object, just a reference to the same model. I used deepcopy to find gradient
"""

from matplotlib.colors import LinearSegmentedColormap
from numpy.random.mtrand import standard_cauchy
from sklearn import cluster
import torch
import sys
from torch._C import _nccl_all_reduce
import network
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
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
from shutil import copyfile
import config
import os
import data_cifar10 as data
from tqdm import tqdm
import tracemalloc

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    print(size, type(obj))
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def noniid_partition(dataset, num_users, lab_per_user, coarse_labels = None):
    """
    Sample non-I.I.D client data
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = lab_per_user*num_users, int(len(dataset)/(lab_per_user*num_users))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    dict_cluster = {}
    dict_cluster_counter = 0
    cluster_id = [i for i in range(num_users)]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, lab_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    if lab_per_user == 2:
        for i in range(num_users):
            a = np.sort([labels[dict_users[i][0]],labels[dict_users[i][num_imgs]]])
            a = (a[0],a[1])

            # print (a)
            if(a in dict_cluster):
                cluster_id[i] = dict_cluster[a]
            else:
                dict_cluster[a] = dict_cluster_counter
                cluster_id[i] = dict_cluster[a]
                dict_cluster_counter = dict_cluster_counter + 1
        dict_cluster_counter +=1

    if lab_per_user == 1:
        cluster_id = []
        for i in range(num_users):
            cluster_id.append(coarse_labels[dict_users[i][0]])
        dict_cluster_counter = len(set(cluster_id))
    # dict_cluster_counter = 1
    # for i in range(len(cluster_id)):
    #     cluster_id[i] = 0
    return dict_users, cluster_id, dict_cluster_counter

class Server():

    def __init__(self, n_c, total_clients, global_lr, model, cluster =0, num_clusters = None, dynamic_clust = False):
        
        self.n_c = n_c
        self.global_lr = global_lr
        self.total_c = total_clients
        self.global_model = model#lenet.LeNet()
        self.global_init_model = copy.deepcopy(model)
        self.clients_alllayer_prevgrad = {}
        self.layers_init = self.global_model.state_dict()
        self.cluster = cluster #1 to implement clustering
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dynamic_clust = dynamic_clust
        
            

        #initialize each layer's gradient to 0 for a client
        for layer in self.layers_init:
            self.layers_init[layer] = torch.zeros_like(self.layers_init[layer], dtype=torch.float64)
        
        if self.cluster == 1:
            self.n_clusters = num_clusters #no of clusters
            print(self.n_clusters, " Clusters")

            #Compute the current cluster center value as a weighted running avg
            self.cluster_center_vals = {}
            for i in range(self.n_clusters):
                self.cluster_center_vals[i] = -1
            

        self.running_av_grads = copy.deepcopy(self.layers_init)
        self.c_i = copy.deepcopy(self.layers_init)
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
    
    def calc_distance(self,vec1, vec2):
        dist = 0
        for layer in vec1:
            dist+= torch.linalg.norm((vec1[layer] - vec2[layer]).view(-1), ord=1)/10
        return dist

    def get_and_update_closest_clust_ind(self,client_id,client_obj,client_update_vect):

        #iterate over cluster centers (distance:l2)
        min_dist = float('inf')
        new_clust_cent = -1
        for clust_key, clust_val in self.cluster_center_vals.items():
            if clust_val == -1:
                self.cluster_center_vals[clust_key] = copy.deepcopy(client_update_vect)
                client_obj[client_id].cluster_center = clust_key
                # print("New cluster found ",client_id, clust_key)
                return clust_key
            # cluster_vect = self.params_to_vec(self.cluster_center_vals[clust_key])
            dist = self.calc_distance(client_update_vect,self.cluster_center_vals[clust_key] )#torch.linalg.norm(client_update_vect - cluster_vect)
            if dist < min_dist:
                min_dist = dist
                new_clust_cent = clust_key

        #Update client's cluster center index to current cluster 
        if new_clust_cent != -1:
            client_obj[client_id].cluster_center = new_clust_cent
            # print("New cluster found ",client_id, min_dist, new_clust_cent)

        # else:
        #     print("Cluster key for client ", client_id, " not updated")
        #     print(dist)
        #     print("clv",client_update_vect)
        #     print("actual ",client_obj[client_id].local_grad_update)

        
        #return closest cluster index
        return new_clust_cent

    
    def get_cluster_prevgrad(self,id,client_obj):
        clust_ind =  client_obj[id].cluster_center
        if self.dynamic_clust:
            if clust_ind != -1:
                return self.cluster_center_vals[clust_ind]
        else:
            if self.cluster_center_vals[clust_ind] != -1:
                return self.cluster_center_vals[clust_ind]

        return copy.deepcopy(self.layers_init)


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

        
    def MIFA(self, ids, client_models, scaff_mods):

        #if self.cluster ==0:
        #sum grads from present clients
            for ii, client in enumerate(ids):
                alllayer_grads = scaff_mods[ii]
                for layer in self.running_av_grads:
                    layer_grad = alllayer_grads[layer].cpu()
                    prevgrad = self.clients_alllayer_prevgrad[client][layer].cpu() 
                    # print("running av",self.running_av_grads[layer].shape)
                    # print("gm",self.global_model.state_dict()[layer].shape)
                    # print("layer-prev",(layer_grad - prevgrad).shape)
                    self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c

                    #self.running_av_grads[layer] += (layer_grad - self.clients_alllayer_prevgrad[client][layer])/self.total_c
                    self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])

            #print(self.running_av_grads[layer])
            return self.running_av_grads
        
        # else:
        #     # sum grads from present clients
        #     for client in ids:
        #         alllayer_grads = client_models[client].local_grad_update
        #         alllayer_prevgrads = self.get_cluster_prevgrad(client,client_models)
        #         if self.cluster_center_vals[0] != -1:
        #             print("\nClient ", client, " gradient norm at cluster 0 is ", self.calc_distance(self.cluster_center_vals[0],self.layers_init))
        #         print("gradient norm at cluster: ",self.calc_distance(alllayer_prevgrads,self.layers_init))
        #         print("gradient norm at client: ",self.calc_distance(alllayer_grads,self.layers_init))

        #         closest_clustcent = self.get_and_update_closest_clust_ind(client,client_models)
        #         for layer in self.running_av_grads:
        #             layer_grad = alllayer_grads[layer].cpu()
        #             prevgrad = alllayer_prevgrads[layer].cpu() 
        #             self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c

        #             #self.running_av_grads[layer] += (layer_grad - self.clients_alllayer_prevgrad[client][layer])/self.total_c
        #             self.cluster_center_vals[closest_clustcent][layer] = copy.deepcopy(alllayer_grads[layer])# (0.2*self.cluster_center_vals[closest_clustcent][layer])+ (1*alllayer_grads[layer])
        #         # print("Client {} belongs to cluster {}".format(client, closest_clustcent))
        #         print("here",closest_clustcent, self.calc_distance(client_models[client].local_grad_update,self.layers_init), self.calc_distance(self.cluster_center_vals[closest_clustcent],self.layers_init) )    
        #     # print(self.running_av_grads[layer])
        #     return self.running_av_grads



    def FedAvg(self, ids, client_models, scaff_mods):
        
        step = copy.deepcopy(self.layers_init)
        #add grads from present clients to running avg
        for ii, client in enumerate(ids):
            alllayer_grads = scaff_mods[ii]
            for layer in alllayer_grads:
                layer_grad = alllayer_grads[layer].cpu()
                step[layer] += layer_grad/self.n_c
            
        return step


    def Scaffold(self, ids, client_models, scaff_mods, delta_c_i):
        
        step = copy.deepcopy(self.layers_init)
        c_i_step = copy.deepcopy(self.layers_init)
        #add grads from present clients to running avg
        for ii, client in enumerate(ids):
            alllayer_grads = scaff_mods[ii]
            alllayer_delta_ci = delta_c_i[ii]
            for layer in alllayer_grads:
                layer_grad = alllayer_grads[layer].cpu()
                layerdeltaci = alllayer_delta_ci[layer].cpu()
                step[layer] += layer_grad/self.n_c
                c_i_step[layer] += layerdeltaci/self.n_c 
            
        return step, c_i_step
      

    def UMIFA(self, ids, client_models, scaff_mods):
        
        umifa_step = copy.deepcopy(self.layers_init)
        running_avg_t = copy.deepcopy(self.running_av_grads)
        #add grads from present clients to running avg
            
        for ii, client in enumerate(ids):
            # sum = 0
            # spr = 0
            alllayer_grads = scaff_mods[ii]
            for layer in self.running_av_grads:
                layer_grad = alllayer_grads[layer].cpu()
                prevgrad = self.clients_alllayer_prevgrad[client][layer].cpu()
                umifa_step[layer] += (running_avg_t[layer] + (layer_grad - prevgrad))/self.n_c
                self.running_av_grads[layer] += (layer_grad - prevgrad)/self.total_c                
                self.clients_alllayer_prevgrad[client][layer] = copy.deepcopy(alllayer_grads[layer])
            #     sum+=np.linalg.norm(prevgrad)
            #     spr+=np.linalg.norm(layer_grad)
            # print(client,"Cluster num: ", "Grad norm: ", sum, spr)
        return umifa_step
        

    def UMIFA_static(self, ids, client_models, scaff_mods):
        clusts_freq = {}
        umifa_step = copy.deepcopy(self.layers_init)
        ii =0
        for client in range(config.total_c):
            if client in ids:
                alllayer_grads = copy.deepcopy(scaff_mods[ii])
                
                if self.dynamic_clust:
                    closest_clustcent = self.get_and_update_closest_clust_ind(client,client_models,copy.deepcopy(scaff_mods[ii]))
                else:
                    closest_clustcent = client_models[client].cluster_center
                    if self.cluster_center_vals[closest_clustcent] == -1:
                        self.cluster_center_vals[closest_clustcent] = copy.deepcopy(self.layers_init)
                ii+=1
                clusts_freq.setdefault(closest_clustcent,0)
                clusts_freq[closest_clustcent]+=1

            else:
                alllayer_grads = self.get_cluster_prevgrad(client,client_models)

            
            # sum = 0
            # spr = 0 
            for layer in alllayer_grads:
                if client in ids:
                    # sum += np.linalg.norm(alllayer_grads[layer].cpu())
                    # spr += np.linalg.norm(self.cluster_center_vals[closest_clustcent][layer]) 
                    alllayer_grads[layer] = self.cluster_center_vals[closest_clustcent][layer] + self.total_c*(alllayer_grads[layer] - self.cluster_center_vals[closest_clustcent][layer])/self.n_c 
                umifa_step[layer] += alllayer_grads[layer].cpu()/self.total_c
            #     sum+=np.linalg.norm(prevgrad)
            #     spr+=np.linalg.norm(layer_grad)
            # if client in ids:
            #     print(client,"Cluster num: ",closest_clustcent, "cluster vs new Grad norm: ", spr, sum)

        for clusts_hit, freq in clusts_freq.items():
            self.cluster_center_vals[clusts_hit] = copy.deepcopy(self.layers_init)
            for ii, client in enumerate(ids):
                if client_models[client].cluster_center == clusts_hit:
                    # sum =0
                    alllayer_grads = copy.deepcopy(scaff_mods[ii])
                    for layer in self.cluster_center_vals[clusts_hit]:
                        self.cluster_center_vals[clusts_hit][layer] += alllayer_grads[layer]/(freq)
                    #     sum += np.linalg.norm(alllayer_grads[layer])
                    # print("clsum", sum)
        return umifa_step


    #ids are participating client IDS
    def aggregate(self, ids, client_models, algo, scaff_mods, delta_c_i):
        
        global_state_dict = self.global_model.state_dict()

        # print(self.global_model.state_dict())
        if algo == 0:
            step = self.MIFA(ids, client_models, scaff_mods)
        elif algo ==1:
            step = self.UMIFA(ids, client_models, scaff_mods)
        elif algo ==2:
            step = self.FedAvg(ids, client_models, scaff_mods)
        elif algo ==3:
            step, delta_c_i_step = self.Scaffold(ids, client_models, scaff_mods, delta_c_i)
            for layer in self.c_i:
                self.c_i[layer] += delta_c_i_step[layer]*(self.n_c/self.total_c)
        elif algo ==4:
            step = self.UMIFA_static(ids, client_models, scaff_mods)
        
        else:
            print("Algo not valid: ", algo)

        

        
        for layer in global_state_dict:
            global_state_dict[layer] = global_state_dict[layer].float()
            global_state_dict[layer] += (step[layer])
        


        self.global_model.load_state_dict(global_state_dict)           


    def test(self, test_data_loader, rnd):
    
        pred =[]
        actuals = []
        l=len(test_data_loader)
        if rnd !=0:
            test_model = copy.deepcopy(self.global_model).cuda()
        else:
            test_model = self.global_init_model.cuda()
        loss =0
        test_model.eval()
        #Over entire test set
        with torch.no_grad():
            for test_X, lab in test_data_loader:
                test_X = test_X.cuda()
                #forward
                out = test_model.forward(test_X)
                
                batch_loss = self.criterion(out,lab.cuda())

                out=torch.argmax(out, axis = 1)
                loss += float(batch_loss.data)
                pred.extend(out)
                actuals.extend(lab)
        loss = loss/l
        # print(pred[0:100], actuals[0:100])
        accuracy = torch.count_nonzero(torch.Tensor(pred)== torch.Tensor(actuals))
        return accuracy/len(pred), loss
    
    
    
def plot_clusters(client_obj_dict, n_c, algo, timestr, num_clusters = config.K):
    if config.clust_mech == 'dynamic':
        final_clusters = np.zeros(config.K)
        bins = list(range(config.K))
    else:
        final_clusters = np.zeros(num_clusters)
        bins = list(range(num_clusters))
    for c in client_obj_dict:
        clust = client_obj_dict[c].cluster_center
        if clust !=-1:
            final_clusters[clust]+=1
    
    plt.title("Distribution of clusters after experiment Algo: "+ config.d_algo[algo])
    plt.bar(bins,final_clusters)#,bins = bins, density = False)
    plt.ylabel('Number of clients')
    plt.xlabel('cluster number')
    if not os.path.exists("./n_c_"+str(n_c)+"/clusthist"):
            #cwd = os.getcwd()
            os.makedirs("./n_c_"+str(n_c)+"/clusthist")
    plt.savefig("./n_c_"+str(n_c)+"/clusthist/"+timestr)
    plt.close()


def round_schedule(lr, rnd, lrfactor, sch_freq):
    if rnd%sch_freq==0 and rnd!=0:
        return lr*lrfactor
    else:
         return lr

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

    def __init__(self, id, lr, cluster_id_i):
        self.id = id
        self.lr = lr
        #model = model 
        #lenet.LeNet()
        # self.model = model
        self.weightDecay = 5e-5
        self.criterion= torch.nn.CrossEntropyLoss()
        
        self.cluster_center = cluster_id_i
        self.local_grad_update = {}
        self.c_i = {}


    
    def train(self,train_data_loader, optimizer, tau,model, lr=0.1, algo = None, c = None, cnt = None):
        avg_loss=0
        #print(self.model.state_dict())
        # print("\n\n\n\n")
        
        model = model.cuda()
        
        
        for i, (train_X, lab) in enumerate(train_data_loader):
            cnt +=1
            # print(lab)
            # break
            #train_X.view(train_X.shape[0],3,32,32).cuda()
            #train_X = train_X.transpose(1,3).cuda() 
            train_X = train_X.cuda()
            lab = lab.cuda()
            optimizer.zero_grad() 
            out=model.forward(train_X.float())
            loss=self.criterion(out,lab)
            loss.backward()
            
            # if algo == 'scaffold':
            #     for layer, p in model.named_parameters():    
            #         self.c_i.setdefault(layer, torch.Tensor([0.0]))
            #         c_layer = torch.Tensor(c[layer].float()).cuda()
            #         c_i_layer = torch.Tensor(self.c_i[layer].float()).cuda()
            #         p.grad = p.grad  +  c_layer -  c_i_layer 
                 
            
            optimizer.step()

            if algo == 'scaffold':
                model_d = model.state_dict()
                for layer, p in model.named_parameters():    
                    self.c_i.setdefault(layer, torch.Tensor([0.0]))
                    c_layer = torch.Tensor(c[layer].float()).cuda()
                    c_i_layer = torch.Tensor(self.c_i[layer].float()).cuda()
                    model_d[layer] = model_d[layer]  -  lr*(c_layer -  c_i_layer) 
                model.load_state_dict(model_d)
            avg_loss+=float(loss.data)

        #     print(i, " Loss inside train: ", loss.data.cpu().numpy())
        # print("/n/n")    
        loss = avg_loss/len(train_data_loader)    
        return loss, model, cnt
    
    def local_train(self,dtrain_loader,lr, model, algo = None, c = None):
        # vec_prev = parameters_to_vector(model.parameters()).cpu()
        oldx = model.state_dict()
        loss =0
        cnt = 0
        model.train()
        delta_c_i = {}
        optimizer = optim.SGD(model.parameters(), lr=lr)#, weight_decay=self.weightDecay)
        #train each client for local epochs
        for epoch in range(config.local_epochs):
            loss_epoch, model_trained, cnt = self.train(dtrain_loader, optimizer, tau, lr = lr, model = model, algo= algo, c=c, cnt = cnt)
            loss+=loss_epoch
        newx = model_trained.cpu().state_dict()
        #print("loss",loss)
        epoch +=1
        if algo == 'scaffold':
            #store y_i in self.local_grad_update 
            local_grad_update = copy.deepcopy(newx)
            
            #store delta y_is in self.local_grad_update
            #also populate c_i and delta c_i
            for layer, val in newx.items(): #torch.Tensor([0.0])#
                new_c_i_layer = self.c_i[layer] - c[layer] + ((oldx[layer]-local_grad_update[layer])/(cnt*lr))
                delta_c_i.setdefault(layer, 0)
                delta_c_i[layer] = new_c_i_layer-self.c_i[layer]
                self.c_i[layer] = new_c_i_layer
                local_grad_update[layer] -= oldx[layer]
            
            return local_grad_update, delta_c_i, loss/config.local_epochs
            #now, delta_y is is stored in self.local_grad_update, delta_c_i in delta_c_i

        else:
            
            local_grad_update = {}
            with torch.no_grad():
                # vec_curr = parameters_to_vector(model_trained.parameters()).cpu()
                # print("Client ", self.id,np.linalg.norm(vec_curr - vec_prev))
                for layer, val in newx.items():
                    local_grad_update.setdefault(layer, 0)
                    local_grad_update[layer] = (newx[layer]-oldx[layer])

        return local_grad_update, loss/config.local_epochs
    
        
    def setWeights(self, global_model_sd):
        self.model.load_state_dict(global_model_sd)


def moving_avg(values):
    smooth_values=[]
    k=1
    lval = len(values) -1
    for i in range(lval+1):
        start = max(0,i-k)
        fin = min(lval, i+k)
        if fin>start:
            smooth_values.append(np.mean(values[start:fin]))
    return values

def plot(loss_algo, acc_algo, test_loss_algo,n_c, timestr, algo_lr, learning_rate):
    # from scipy.interpolate import make_interp_spline, BSpline
    #plot train loss (local)
    loss_algo = moving_avg(loss_algo)
    acc_algo = moving_avg(acc_algo)
    test_loss_algo = moving_avg(test_loss_algo)

    plt.figure(figsize=(10,7)) 
    plt.ylabel('Loss')


    
    plt.xlabel('Number of Comm rounds')
    for algo_i, loss_per_algo in enumerate(loss_algo):
        plt.title('Accuracy vs Comm rounds lr = {0:.5f}, clients = {1}/{2}, lr_decay = {3}, local_ep = {4} '.format(
            algo_lr[algo_i][learning_rate],n_c, config.total_c, config.lrfactor[algo_i], config.local_epochs))  
        # xo = np.arange(len(loss_per_algo))*config.plot_every_n
        # xnew = np.linspace(xo.min(), xo.max(), 600)
        # spl = make_interp_spline(xo, loss_per_algo) 
        # y_smooth = spl(xnew)
        # plt.plot(xnew, y_smooth,  label = d_algo[d_algo_keys[algo_i]])
        plt.plot(np.arange(len(loss_per_algo))*config.plot_every_n, loss_per_algo,  label = d_algo[d_algo_keys[algo_i]])
        # plt.plot(, loss_per_algo)
    plt.legend(loc = 'upper right')
    dir_name ="./n_c_"+str(n_c)+"/train/{0}.png".format(timestr)
    if not os.path.exists("./n_c_"+str(n_c)+"/train"):
        #cwd = os.getcwd()
        os.makedirs("./n_c_"+str(n_c)+"/train")
    plt.savefig(dir_name)

    #plot accuracy test 
    plt.figure(figsize=(10,7)) 
    plt.ylabel('Test Accuracy')
    plt.xlabel('Number of Comm rounds')
    for algo_i, acc_per_algo in enumerate(acc_algo):
        plt.title('Accuracy vs Comm rounds lr = {0:.5f}, clients = {1}/{2}, lr_decay = {3}, local_ep = {4} '.format(
            algo_lr[algo_i][learning_rate],n_c, config.total_c, config.lrfactor[algo_i], config.local_epochs)) 
        # xo = np.arange(len(acc_per_algo))*config.plot_every_n
        # xnew = np.linspace(xo.min(), xo.max(), 600)
        # spl = make_interp_spline(xo, acc_per_algo) 
        # y_smooth = spl(xnew)
        plt.plot(np.arange(len(acc_per_algo))*config.plot_every_n, acc_per_algo,  label = d_algo[d_algo_keys[algo_i]])
        # plt.plot(xnew, y_smooth,  label = d_algo[d_algo_keys[algo_i]])
        #plt.plot(np.arange(len(acc_per_algo))*config.plot_every_n, acc_per_algo, label = d_algo[d_algo_keys[algo_i]])
    plt.legend(loc = 'lower right')
    dir_name ="./n_c_"+str(n_c)+"/test/{0}.png".format(timestr)
    if not os.path.exists("./n_c_"+str(n_c)+"/test"):
        #cwd = os.getcwd()
        os.makedirs("./n_c_"+str(n_c)+"/test")
    plt.savefig(dir_name)

    #plot loss test 
    plt.figure(figsize=(10,7)) 
    plt.ylabel('Test Loss')
    plt.xlabel('Number of Comm rounds')
    for algo_i, tstloss_per_algo in enumerate(test_loss_algo):
        plt.title('Accuracy vs Comm rounds lr = {0:.5f}, clients = {1}/{2}, lr_decay = {3}, local_ep = {4} '.format(
            algo_lr[algo_i][learning_rate],n_c, config.total_c, config.lrfactor[algo_i], config.local_epochs))    
        # xo = np.arange(len(tstloss_per_algo))*config.plot_every_n
        # xnew = np.linspace(xo.min(), xo.max(), 600)
        # spl = make_interp_spline(xo, tstloss_per_algo) 
        # y_smooth = spl(xnew)
        # plt.plot(xnew, y_smooth,  label = d_algo[d_algo_keys[algo_i]])
        plt.plot(np.arange(len(tstloss_per_algo))*config.plot_every_n, tstloss_per_algo,  label = d_algo[d_algo_keys[algo_i]])
        #plt.plot(np.arange(len(tstloss_per_algo))*config.plot_every_n, tstloss_per_algo, label = d_algo[d_algo_keys[algo_i]])
    plt.legend(loc = 'upper right')
    dir_name ="./n_c_"+str(n_c)+"/test_loss/{0}.png".format(timestr)
    if not os.path.exists("./n_c_"+str(n_c)+"/test_loss"):
        #cwd = os.getcwd()
        os.makedirs("./n_c_"+str(n_c)+"/test_loss")
    plt.savefig(dir_name)
    #plt.close()

class DatasetSplit(Dataset):
    def __init__(self, idxs):
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = dataset_train[self.idxs[item]]
        return image, label

#To add transforms for EMNIST train and test datset
class DatasetEMNIST(Dataset):
    def __init__(self, dataset,transforms, dict_users = None):
        new_dict_users = {}
        if dict_users:
            self.dataset = []
            i=0
            for k, v in dict_users.items():
                new_dict_users[k] = []     
                for value in v:
                    self.dataset.append(dataset[value])
                    new_dict_users[k].append(i)
                    i+=1
        else:
            self.dataset = dataset
        self.transforms = transforms
        self.dict_users = new_dict_users
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return self.transforms(image), label


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    np.random.seed(10)
    #Init hyperparams
    client_names=list(range(config.total_c))
    batch_size=config.batch_size
    n_rnds=config.n_rnds
    tau=config.tau
    total_c=config.total_c
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
    if config.dataset == 'cifar10':
        trans_cifar_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        trans_cifar_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        dict_users, cluster_id, num_clusters = noniid_partition(dataset_train, config.total_c, 2)

    elif config.dataset == 'cifar100':
        trans_cifar_train = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trans_cifar_test = transforms.Compose([transforms.ToTensor(), \
            #transforms.CenterCrop(24),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        import pickle
        train_file = 'data/cifar100/cifar-100-python/train'
        with open(train_file, 'rb') as fo:
            train_datadict = pickle.load(fo, encoding='bytes')
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar_test)
        dict_users, cluster_id, num_clusters = noniid_partition(dataset_train, config.total_c, 1, coarse_labels = train_datadict[b'coarse_labels'])
      

    elif config.dataset == 'emnist':        
        emnist_data = np.load('data/emnist_dataset_umifa.npy', allow_pickle= True).item()
        dataset_train = emnist_data['dataset_train']
        dataset_test = emnist_data['dataset_test']
        dict_users_emnist = emnist_data['dict_users']
        emnist_clients = list(dict_users_emnist.keys())
        #Pick config.no_of_c number of random clients from the total number of clients in dataset
        client_names=list(range(config.total_c))
        total_c = len(client_names)

        chosen_clients = np.random.choice(emnist_clients, total_c)
        dict_users = {client_names[i]:dict_users_emnist[chosen_clients[i]] for i in range(total_c)}
        transform =  transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.RandomRotation(90),
                        transforms.Normalize((0.1307,), (0.3081,))])
        trans_test = transforms.Compose([
            transforms.ToPILImage(),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.RandomRotation(90),
                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = DatasetEMNIST(dataset_train, transform, dict_users)
        dict_users = dataset_train.dict_users
        dataset_test = DatasetEMNIST(dataset_test, trans_test)

    
    elif config.dataset == 'shakespeare':     
        ALL_LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        NUM_LETTERS = len(ALL_LETTERS)   
        shakespeare_data = np.load('data/shakespeare/shakespeare_data_umifa.npy', allow_pickle= True).item()
        user_to_play_map =  np.load('data/shakespeare/user_to_play_map.npy', allow_pickle= True).item()
        clust_num = set([])
        for k,v in user_to_play_map.items():
            clust_num.add(v)
        num_clusters = len(clust_num)
        cluster_id = user_to_play_map
        dataset_train = shakespeare_data['dataset_train']
        dataset_test = shakespeare_data['dataset_test']
        dict_users = shakespeare_data['dict_users']
        print(len(dict_users.keys()), "clients")
        # emnist_clients = list(dict_users_emnist.keys())
        #Pick config.no_of_c number of random clients from the total number of clients in dataset
        # client_names=list(range(config.total_c))
        # total_c = len(client_names)

        # chosen_clients = np.random.choice(emnist_clients, total_c)
        # dict_users = {client_names[i]:dict_users_emnist[chosen_clients[i]] for i in range(total_c)}
        # dataset_train = DatasetEMNIST(dataset_train, transform, dict_users)
        # dict_users = dataset_train.dict_users
        # dataset_test = DatasetEMNIST(dataset_test, trans_test)

    else:
        print("Please specify dataset")
        raise ValueError("Invalid dataset argument")

    if config.cluster ==1:
        if config.clust_mech == 'dynamic':
            print("Dynamic clustering")
            num_clusters = config.K
            cluster_id = None
            dynamic_clust = 1
        else:
            print("Static clustering")
            dynamic_clust =0
    else:
        dynamic_clust = 0
    d_algo = config.d_algo
    d_algo_keys = sorted(list(d_algo.keys()))
    lrfactor = config.lrfactor
    sch_freq = config.sch_freq
    client_data_split_dict = {}
    pi = []
    for i in list(range(config.total_c)):
        if int(i/(10))%2==0:
            pi.append(int(i/(10)))
        else:
            pi.append(int(i/(10))-1)

    pi = (np.array(pi)*config.pi_min/9) + (1-config.pi_min)
    
    for c in client_names:
        client_data_split_dict[c] = DatasetSplit(dict_users[c])
    # 0 corresponds to mifa
    # 1 corresponds to umifa
    # 2 corresponds to fedavg
    if config.model_type == 'r':
        print("Resnet18")
        cmodel = network.resnet18()

    elif config.model_type == 'cnnmnist':
        print("network: CNN MNIST")
        cmodel = network.CNNMnist()
    elif config.model_type == 'lenetmnist': 
        print("LeNetmnist")
        cmodel = network.LeNetmnist()
    elif config.model_type == 'mlp': 
        cmodel = network.MLP()
        print("mlp")
    elif config.model_type == 'lenet': 
        print("LeNet")
        cmodel = lenet.LeNet()
    elif config.model_type == 'shakespeare_lstm':
        print('Shakespeare LSTM')
        tempmodel = network.ShakespeareLstm(num_classes=NUM_LETTERS, default_batch_size = config.batch_size)
        #Deepcopying this model results in errors
        cmodel = network.ShakespeareLstm(num_classes=NUM_LETTERS, default_batch_size = config.batch_size)
        cmodel.load_state_dict(tempmodel.state_dict())
        a = copy.deepcopy(cmodel)
    else:
        print('please choose a network')

    for choose_nc in config.no_of_c:

        loss_eachlr=[] 
        acc_eachlr=[]
        test_loss_eachlr=[]
        global_loss_eachlr = []

        for learning_rate in range(len(config.algo_lr[0])):
            timestr = time.strftime("%Y%m%d-%H%M%S")            
            loss_algo=[] 
            acc_algo =[] 
            test_loss_algo =[]
            global_train_loss_algo = []
            if not os.path.exists("./n_c_"+str(choose_nc)+"/configfile"):
            #cwd = os.getcwd()
                os.makedirs("./n_c_"+str(choose_nc)+"/configfile")
            copyfile("config.py", "./n_c_"+str(choose_nc)+"/configfile/"+timestr)
            if config.sel_client_variablepi == True:
                print("Varible number of clients selected per round")
                idxs_users_allrounds = []
                for r in range(config.n_rnds):
                    clients_participating_bool = []
                    for c in range(len(pi)):
                        clients_participating_bool.append(np.random.choice([0,1], p = [1-pi[c],pi[c]]))
                    p_clients = [ i for i,v in enumerate(clients_participating_bool) if v==1]
                    idxs_users_allrounds.append(np.array(p_clients))
            else:
                idxs_users_allrounds = [np.random.choice(client_names,choose_nc,replace = False) for i in range(config.n_rnds)]
            #uncommented
            # idxs_users_allrounds = [np.random.choice([9,4],2,replace = False) for i in range(config.n_rnds)]

            #Run simulation for each algorithm
            for algo in list(d_algo.keys()): 

                global_lr = 1
                lr = config.algo_lr[algo][learning_rate]
                print("--------------------------Algo: {0}------------------------------------".format(d_algo[algo]))     
                
                server = Server(choose_nc, total_c, global_lr, model = copy.deepcopy(cmodel),cluster = config.cluster, num_clusters= num_clusters, dynamic_clust = dynamic_clust)
                client_object_dict = {}

                for c in client_names:
                    if cluster_id:
                        cluster_id_i = cluster_id[c]
                    else:
                        cluster_id_i = -1
                    client_object_dict[c]= Client(c, lr, cluster_id_i)
                
                local_rnd_loss=[]
                local_rnd_acc=[]
                rnd_test_loss = []
                rnd_train_loss = []

                
                #Global epochs
                for rnd in (range(config.n_rnds)): # each communication round
                    size = 0
                    # for c in client_object_dict:
                    #     cs = get_size(client_object_dict[c])
                    #     size +=cs
                    #     print("Client " ,c, " ", cs)
                    # print("total size ", size)
                    #schedule(server, mode = 'global')
                    #schedule()
                    lr = round_schedule(lr, rnd, lrfactor[algo], sch_freq)
                    if rnd ==0 and config.enforce_cp_r1 ==1 and algo ==0:
                        print("Running all clients in first round")
                        idxs_users = client_names
                    else:
                        idxs_users=idxs_users_allrounds[rnd]
                    # print("chosen clients",idxs_users)
                    idxs_len=len(idxs_users)
                    # #Obtain global weights
                    # global_state_dict=server.global_model.state_dict()

                    # #Assign each client model to global model
                    # for c in idxs_users:  
                    #     client_object_dict[c].setWeights(global_state_dict)


                    d_train_losses=0 #loss over all clients
                    scaff_mods = []
                    delta_c_i_l = []
                    for sel_idx in idxs_users:

                        #Get train loader
                        dtrain_loader = DataLoader(client_data_split_dict[sel_idx], batch_size=config.batch_size, shuffle=True)
                        if algo==3:
                            local_grad_up, delta_c_i, loss = client_object_dict[sel_idx].local_train(dtrain_loader, lr = lr,  model = copy.deepcopy(server.global_model), algo = d_algo[algo], c = copy.deepcopy(server.c_i))
                            scaff_mods.append(local_grad_up)
                            delta_c_i_l.append(delta_c_i)
                        else:
                            local_grad_up, loss = client_object_dict[sel_idx].local_train(dtrain_loader, lr = lr,  model = copy.deepcopy(server.global_model), algo = d_algo[algo], c = copy.deepcopy(server.c_i))
                            scaff_mods.append(local_grad_up)
                            # print(loss)

                        
                        d_train_losses+=loss
                    d_train_losses /= idxs_len

                    
                    #aggregate at server
                    if algo==3:
                        server.aggregate(idxs_users, client_object_dict, algo, scaff_mods, delta_c_i_l)
                    else:
                        server.aggregate(idxs_users, client_object_dict, algo, scaff_mods , None)
                    
                    #Uncomment 
                    
                    # Calculate test acc and loss every plot_every_n epochs
                    if rnd%config.plot_every_n == 0: 
                        dtest_loader = DataLoader(dataset_test,batch_size = config.batch_size, shuffle = False)
                        dtrain_global_loader = DataLoader(dataset_train,batch_size = config.batch_size, shuffle = False)
                        acc, val_loss = server.test(dtest_loader, rnd)
                        if not config.plot_local_train_loss:  
                            global_train, global_train_loss = server.test(dtrain_global_loader)
                            rnd_train_loss.append(global_train_loss)
                            print("Global train loss: ", global_train_loss)

                        local_rnd_acc.append(acc)
                        local_rnd_loss.append(d_train_losses)
                        rnd_test_loss.append(val_loss)
                        print("Testing acc: ",acc, "Test loss: ", val_loss)
                        
                        
                    # if rnd%100 == 1:
                    #     print("collecting")
                    #     gc.collect()
                    #     print("collected")
                    

                    # if(rnd==100):
                    #     lr = lr/2


                    print("Rnd {4} - Training Error: {0}, Algo: {2}, n_c: {3}, lr: {1}\n".format(d_train_losses,lr, algo, choose_nc, rnd))
                    # #mem leak
                    # snapshot = tracemalloc.take_snapshot()
                    # top_stats = snapshot.statistics('traceback')

                    # # pick the biggest memory block
                    # stat = top_stats[0]
                  

                loss_algo.append(local_rnd_loss)  
                acc_algo.append(local_rnd_acc) 
                test_loss_algo.append(rnd_test_loss)
                global_train_loss_algo.append(rnd_train_loss)
                plot_clusters(client_object_dict, choose_nc,algo, timestr, num_clusters = num_clusters)


                


            loss_eachlr.append(loss_algo)
            acc_eachlr.append(acc_algo)
            test_loss_eachlr.append(test_loss_algo)
            global_loss_eachlr.append(global_train_loss_algo)
            #record final values of loss and acc
            file_object = open("./n_c_"+str(choose_nc)+"/configfile/"+timestr, 'a')
            file_object.write("loss_algo = " + str(loss_algo)+"\n")
            file_object.write("acc_algo = " +str(acc_algo)+"\n")
            file_object.write("test_loss_algo= "+str(test_loss_algo)+"\n")
            file_object.write("global_train_loss_algo = "+str(global_train_loss_algo)+"\n")
            file_object.close()
            if config.plot_local_train_loss:
                global_train_loss_algo = loss_algo
            plot(global_train_loss_algo, acc_algo, test_loss_algo,choose_nc, timestr, config.algo_lr, learning_rate)            
        

                
                
                    

