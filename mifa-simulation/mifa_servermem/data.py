#!/usr/bin/env python
# coding: utf-8
import os
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torch.utils.data as data_utils
import struct
import matplotlib.pyplot as plt
import config

#Define your data path (the directory containing the 4 np array files)
DATA_PATH = '/home/aushim/Desktop/fedmifa/mifa-simulation/mifa_clientmem/'

class MNIST(Dataset):
    def __init__(self, set_name):
        super(MNIST, self).__init__()
        if set_name=='train':
            
            with open(DATA_PATH+'train-labels.idx1-ubyte','rb') as f:
                nrows, ncols = struct.unpack(">II", f.read(8))
                load_y = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
               

            with open(DATA_PATH+'train-images.idx3-ubyte','rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                nrows, ncols = struct.unpack(">II", f.read(8))
                data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
                load_x = data.reshape((size, nrows*ncols))
                
        else:
            with open(DATA_PATH+'t10k-images.idx3-ubyte','rb') as f:
                magic, size = struct.unpack(">II", f.read(8))
                nrows, ncols = struct.unpack(">II", f.read(8))
                data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
                load_x = data.reshape((size, nrows*ncols))

            with open(DATA_PATH+'t10k-labels.idx1-ubyte','rb') as f:
                nrows, ncols = struct.unpack(">II", f.read(8))
                load_y = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
      
        self.x=load_x.astype('float64')/255
        self.y=np.array(load_y)
        if set_name=='train': 
            self.client_partition={}
            self.split()
            self.index_to_data()


    def split(self,n_clients=100, n_targets=10): #60000 samples, 10 classes, each client gets 600 samples - 300 of one class and 300 of another 
        #data_part_by_class ={0:[0,1,2,3,4],1:[5,6,7,8,9]... }
        data_part_by_class={}
        
        #create dictionary to hold data based on class
        for i in range(n_targets):
            data_part_by_class[i]=[]
        for i in range(len(self.y)):            
            data_part_by_class[int(self.y[i])].append(i)

        samples_per_client = len(self.y)/n_clients
        samplesper_clientper_class=samples_per_client/2 #2 because each client has only data from 2 classes here
        #create client partitioned data
        target=0
        c_i=0
        sli1=300
        sli2=300
        next_target=target+1
        while (c_i <=n_clients and target<10):
            self.client_partition[c_i]=[]
            if(target ==9):
                self.client_partition[c_i].extend(data_part_by_class[target][0:600])
                data_part_by_class[next_target]=data_part_by_class[target][600:]
                c_i+=1
                if c_i ==100:
                    break
                continue
            if(len(data_part_by_class[target])>=samplesper_clientper_class and len(data_part_by_class[next_target])>=samplesper_clientper_class):      
                self.client_partition[c_i].extend(data_part_by_class[target][0:sli1])
                self.client_partition[c_i].extend(data_part_by_class[next_target][0:sli2])
                data_part_by_class[target]=data_part_by_class[target][sli1:]
                data_part_by_class[next_target]=data_part_by_class[next_target][sli2:]
                c_i+=1
                
                sli1=sli2=300 #how much to slice next round

                if (len(data_part_by_class[target])==samplesper_clientper_class):
                    if (len(data_part_by_class[next_target])==samplesper_clientper_class):
                        target=next_target+1
                        next_target+=1
                    else:
                        target=next_target

                if (len(data_part_by_class[next_target])==samplesper_clientper_class):
                    next_target +=1

                continue


            #if first class has enough samples but second doesnt
            elif (len(data_part_by_class[target])>=samplesper_clientper_class and len(data_part_by_class[target])+len(data_part_by_class[next_target])>=600):
                self.client_partition[c_i].extend(data_part_by_class[target][0:int((2*samplesper_clientper_class)-len(data_part_by_class[next_target]))])
                self.client_partition[c_i].extend(data_part_by_class[next_target][0:]) #take max samples from 1st class and remaining from 2nd
                data_part_by_class[target]=data_part_by_class[target][int(2*samplesper_clientper_class)-len(data_part_by_class[next_target]):]
                data_part_by_class[next_target]=[]
                c_i+=1
                sli1= sli2=300
                next_target +=1
                continue

            #if second class has enough samples but first doesnt
            elif (len(data_part_by_class[next_target])>=samplesper_clientper_class and len(data_part_by_class[target])+len(data_part_by_class[next_target])>=600):
                self.client_partition[c_i].extend(data_part_by_class[target][0:])
                self.client_partition[c_i].extend(data_part_by_class[next_target][0:int((2*samplesper_clientper_class)-len(data_part_by_class[target]))]) #take max samples from 2nd class and remaining from 1st
                data_part_by_class[target]=[]
                data_part_by_class[next_target]=data_part_by_class[next_target][int((2*samplesper_clientper_class)-len(data_part_by_class[target])):]
                c_i+=1
                sli1= sli2=300
                target=next_target
                next_target=target+1

                
                continue

            #both dont have enough samples and dont sum to 600
            #use up samples from both and then from the next class
            else: 
                self.client_partition[c_i].extend(data_part_by_class[target][0:len(data_part_by_class[target])])
                self.client_partition[c_i].extend(data_part_by_class[next_target][0:len(data_part_by_class[next_target])]) #take max samples both classes
                sli1 = 600-(len(data_part_by_class[target])+len(data_part_by_class[target+1]))
                data_part_by_class[target]=[]
                data_part_by_class[next_target]=[]               
                sli1= sli2=300
                target=next_target+1
                next_target+=2
                self.client_partition[c_i].extend(data_part_by_class[target][0:sli1])
                c_i+=1      
            
                
                    
        
    
    def index_to_data(self,n_clients=100):
        self.dataset={}
        for c_i in range(n_clients):
            self.dataset[c_i]={'x':[],'y':[]}
            for d_i in self.client_partition[c_i]:
                self.dataset[c_i]['x'].append(self.x[d_i])
                self.dataset[c_i]['y'].append(self.y[d_i])

    #If client selection paradigm is set to setting pi by label
    def selection_prob(self, n_clients=100):
        probs=[]
        pmin=config.pi_min
        for c_i in range(n_clients):
            j = self.dataset[c_i]['y'][0]
            p_i = (pmin * j/9) + (1-pmin)  #pmin is 0.1
            probs.append(p_i)
        return probs

          
    def __len__(self):
        return len(self.x)
    
        
def init_dataset():
    train_data_obj=MNIST('train')
    test_data_obj=MNIST('test')
    p_i = train_data_obj.selection_prob()

    return train_data_obj,test_data_obj, p_i


def get_train_data_loader(train_data_obj, client_i,batch_size):
    train_data = data_utils.TensorDataset(torch.tensor(train_data_obj.dataset[client_i]['x']).type(torch.DoubleTensor), torch.tensor(train_data_obj.dataset[client_i]['y']).type(torch.LongTensor))  
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
   
   
    return train_dataloader


def get_test_data_loader(test_data_obj, batch_size =1):

    test_data = data_utils.TensorDataset(torch.tensor(test_data_obj.x).type(torch.FloatTensor), torch.tensor(test_data_obj.y).type(torch.LongTensor))    
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle=False)
   
    return test_dataloader
