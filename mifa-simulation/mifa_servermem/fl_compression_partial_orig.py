# -*- coding: utf-8 -*-
"""FL_Compression_Partial (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OdKdDOUtxBsLjez4MbHk0xPY-Hg708Uw
"""

import numpy as np
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
import scipy
from torchvision import datasets, transforms
#from torchsummary import summary
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn
import torch.nn.functional as F
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy
import random
from sklearn import metrics
import torch.nn.functional as func

import seaborn as sns
sns.set()

args = {
    'iid' : 0,
    'num_users' : 500,
    'device' : 'cuda',
    'frac' : 1.0,
    'local_bs':64,
    'local_ep':5,
    'momentum':0,
    'epochs':300,
    'bs':100,
    'verbose':0,
    'dataset':'cifar',
    'num_channels':3,
    'num_classes':10,
}

#################################################
# Dataloader routines
################################################
def iid_partition(dataset, num_users):
    """
    Sample I.I.D. client data
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


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



# load dataset and split users
if args['dataset'] == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    # sample users
    if args['iid']:
        dict_users = iid_partition(dataset_train, args['num_users'])
    else:
        dict_users = noniid_partition(dataset_train, args['num_users'])

elif args['dataset'] == 'fmnist':
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST('./data/fmnist/', train=True, download=True, transform=trans_fmnist)
    dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=trans_fmnist)
    # sample users
    if args['iid']:
        dict_users = iid_partition(dataset_train, args['num_users'])
    else:
        dict_users = noniid_partition(dataset_train, args['num_users'])

elif args['dataset'] == 'cifar':
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
    if args['iid']:
        dict_users = iid_partition(dataset_train, args['num_users'])
    else:
        dict_users = noniid_partition(dataset_train, args['num_users'])

else:
    exit('Error: unrecognized dataset')

#####################################################
# Neural net architectures
####################################################


class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 10)
        # self.linear2 = nn.Linear(128, 64)
        # self.linear3 = nn.Linear(128,output_size)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear1(x))
        x = self.linear1(x)

        return(x)

class CNNCifar(nn.Module):
    def __init__(self,args):
        super(CNNCifar, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 48, 3)
        self.conv1_bn = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(48, 48, 3)
        self.conv2_bn = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 48, 3)
        self.conv3_bn = nn.BatchNorm2d(48)

        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(48*4*4, 256)
        self.fc3 = nn.Linear(256, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.conv1_bn(x)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.conv2_bn(x)
        # print (x.shape)
        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)

        # print (x.shape)

        x = x.view(-1, 48*4*4)
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x





class CNNMnist(nn.Module):
    def __init__(self,args):
        super(CNNMnist, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 6*6 from image dimension
        self.fc2 = nn.Linear(128, 10)
        self.ceriation = nn.CrossEntropyLoss()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# NN from the Federated Learining AISTATS paper
class FLNet(nn.Module):
    def __init__(self,args):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(args['num_channels'], 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5,padding=2)
        # self.bn1 = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(64*8*8, 512)  
        # self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, args['num_classes'])

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = self.bn1(x)
        # print (x.shape)
        x = x.view(-1,  x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        # x = self.bn2(x)
        x = self.fc2(x)
        return x

###############################################
# Routine for computing test accuracy
##############################################
def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args['verbose']:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_loss_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

    test_loss /= len(data_loader.dataset)

    if args['verbose']:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return test_loss


##############################################
# Client side optimization
##############################################
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args['local_bs'], shuffle=True)
        
    def train_and_sketch(self, net, d, eta, subsample, m):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=eta, momentum=self.args['momentum'])
        # print(doSketching)
        epoch_loss = []
        prev_net = copy.deepcopy(net)
        for iter in range(self.args['local_ep']):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                # break

            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            
        # Sparsify the difference between previous and current model
        with torch.no_grad():
            if subsample < d: 
                
                # Comoute the difference between previous and current model
                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr - vec_prev

                
                params_delta_vec_np = params_delta_vec.cpu().numpy()
                model_to_return = params_delta_vec_np

            else:
                # No subsampling
                model_to_return = net.state_dict()
            
        return model_to_return, sum(epoch_loss) / len(epoch_loss)

#######################################
# Compute summation routine to be used at the server
#######################################
def FedAvg_sketch(w):
    w_sum = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        w_sum += w[i]
    return w_sum

def active_clients(n,p):
  
  # l = range(n)
  a = np.random.choice(n,int(n*p[0]),replace=False)

  # a =[]
  # for i in range(n):
  #   r = np.random.uniform(0,1)
  #   # print (r)
  #   if(r<p[i]):
  #     a.append(i)

  return a

def node_select(vect,pos,ind):

  v = np.zeros((d,))
  if (pos in ind):
    return vect
  else:
    return v


def avg(vect_list):
    return sum(vect_list)/len(vect_list)

    
def fed_avg_bias_decoder(comp_list,d,n,ind):


  est_vect = np.zeros((d,))

  for i in ind:
    est_vect += comp_list[i]

  a = len(ind)
  est_vect = est_vect/a

  return est_vect

def fed_avg_unbias_decoder(comp_list,d,n,ind,p):


  est_vect = np.zeros((d,))

  for (j,i) in enumerate(ind):
    if(len(ind)==n):
      est_vect += comp_list[j]
    else:
      est_vect += comp_list[j]/p[i]

  a = len(ind)
  est_vect = est_vect/n

  return est_vect



def fed_avg_mem_bias_decoder(comp_list,d,n,mem_mat,ind):

  for (j,i) in enumerate(ind):
    mem_mat[i] = comp_list[j]


  s = np.sum(mem_mat,axis=0)

  est_vect = s/n


  return est_vect


def fed_avg_mem_bias_adap_decoder(comp_list,d,n,mem_mat,ind):

  for (j,i) in enumerate(ind):
    mem_mat[i] = comp_list[j]


  s = np.sum(mem_mat,axis=0)

  c = np.count_nonzero(mem_mat[:,0])  ##acts as a counter for the number of non-zero gradients in server memory (i.e. number of unique clients that have participated)

  # print(c)
  est_vect = s/c


  return est_vect

def fed_avg_uncompressed(comp_list,n,mem_mat):
  for i in range(n):
    mem_mat[i] = comp_list[i]

  v_dec = avg(comp_list)
  return v_dec


def fed_avg_mem_unbias_decoder(comp_list,d,n,mem_mat,ind,p):
 
  est_vect = np.zeros((d,))
  
  x = np.zeros((d,))
  v = np.zeros((d,))

  s = np.sum(mem_mat,axis=0)
 
  for (j,i) in enumerate(ind):

    if(len(ind)==n):
      v += comp_list[j]
    else:
      v += (comp_list[j] - mem_mat[i])/p[i]
    mem_mat[i] = comp_list[j]


 
  est_vect = (s+v)/n


  return est_vect

def comp_func(vect_list,d,n,p,ind,comp_type):

  
  if(comp_type=='Uncompressed'):
      v_dec = avg(vect_list)
      err = 0

      return v_dec, err


  if (comp_type=="FedAvg(Biased)"):

    v_dec = fed_avg_bias_decoder(vect_list,d,n,ind)

    # err = np.linalg.norm(v_dec-avg(vect_list))**2
    err = 0

    return v_dec,err

  if (comp_type=="FedAvg(Unbiased)"):

    v_dec = fed_avg_unbias_decoder(vect_list,d,n,ind,p)


    # err = np.linalg.norm(v_dec-avg(vect_list))**2
    err = 0

    return v_dec,err


def comp_func_mem(vect_list,d,n,p,mem_mat,v,ind,comp_type,t):
 
 

    
    if(comp_type=='Uncompressed'):
      v_dec = fed_avg_uncompressed(vect_list,n,mem_mat)
      err = 0

      return v_dec, err
    
    if (comp_type == "MIFA"):


      v_dec = fed_avg_mem_bias_decoder(vect_list,d,n,mem_mat,ind)

      err = 0
 
      return v_dec, err


    if (comp_type == "MIFA(Adaptive)"):


      v_dec = fed_avg_mem_bias_adap_decoder(vect_list,d,n,mem_mat,ind)



      # err = np.linalg.norm(v_dec-avg(vect_list))**2
      err = 0
 
      return v_dec, err
 

 
    if(comp_type == "U-MIFA"):

        v_dec = fed_avg_mem_unbias_decoder(vect_list,d,n,mem_mat,ind,p)

        # err = np.linalg.norm(v_dec-avg(vect_list))**2
        err = 0
 
        return v_dec, err

loss_train = {}
test_acc_epochs = {}
loss_train_epochs = {}
eta = {}

mem_methods = ["MIFA","U-MIFA","MIFA(Adaptive)"]

n = args['num_users']

p = np.ones((n,))

for i in range(n):
  p[i] = 0.01

##################################
### Main: Server side optimization and testing
#################################
# Initiate the NN


s0 = 'FedAvg(Unbiased)'
s1 = 'MIFA'
s2 = 'U-MIFA'
s3 = 'MIFA(Adaptive)'


eta[s0] = 0.1
eta[s1] = 0.1
eta[s2] = 0.1
eta[s3] = 0.1


compression_methods = [s1]

# args['epochs'] = 500



# net_glob_org = MLP_CIFAR(32*32*3,10).to(args['device'])
# net_glob_org = MLP(28*28,10).to(args['device'])
net_glob_org = LeNet().to(args['device'])

for compression in compression_methods:

    print (compression)
    net_glob = copy.deepcopy(net_glob_org)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()



    d = sum(p.numel() for p in net_glob.parameters() if p.requires_grad)
    print (d)
    subsample = 1


    
    m = args['num_users']
    mem_mat = np.zeros((m,d))
    v = np.zeros((1,d))
    loss_train_temp = []
    loss_train_epochs_temp = []
    test_acc_temp = []

    for i in range(args['epochs']):
        

        w_locals, loss_locals, mask_locals = [], [], []
        epoch_err = 0

        if(i==-1):
          ind = range(n)
        else:
          ind = active_clients(n,p)

                
        for idx in ind:

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_model, loss = local.train_and_sketch(copy.deepcopy(net_glob), d, eta[compression],subsample, m)
                    
            w_locals.append(copy.deepcopy(w_model))
            loss_locals.append(copy.deepcopy(loss))
            

        with torch.no_grad():

            if(compression in mem_methods):

              w_avg,err = comp_func_mem(w_locals,d,n,p,mem_mat,v,ind,compression,i)

            else:
              w_avg,err = comp_func(w_locals,d,n,p,ind,compression)

            epoch_err +=err
            w = torch.tensor(w_avg).type(torch.FloatTensor)


                
            if subsample < d:
                w_vec_estimate = parameters_to_vector(net_glob.parameters()) + w.to(args['device'])
                vector_to_parameters(w_vec_estimate,net_glob.parameters())
            else:
                net_glob.load_state_dict(w_avg)
                
        # print loss
        # print (loss_locals)
        loss_avg = sum(loss_locals) / len(loss_locals)
        epoch_err = epoch_err
        print('Round ',i, 'Round loss',loss_avg)
        # print('Round {:3d}, Average error {:.3f}'.format(i, epoch_err))
        loss_train_temp.append(loss_avg)
        # err_epochs_temp.append(epoch_err)

        # net_glob.eval()
        

        if(i%100==0):
          loss_train_2 = test_loss_img(net_glob, dataset_train, args)
          print('Round ',i, 'Objective loss',loss_train_2)
          acc_test, loss_test_2 = test_img(net_glob, dataset_test, args)
          print("Round ",i,"Test Accuracy",acc_test)


          test_acc_temp.append(acc_test)
          loss_train_epochs_temp.append(loss_train_2)



    loss_train[compression] = loss_train_temp
    loss_train_epochs[compression] = loss_train_epochs_temp
    test_acc_epochs[compression] = test_acc_temp

start_pos = 0

# r = 100*[i for i in range(1,len(loss_train_epochs['U-MIFA'][0:]))]
r = [100*i for i in range(0,int(args['epochs']/100))]



for key in loss_train:
  # if(key!='Uncompressed'):
    plt.plot(r,loss_train_epochs[key][start_pos:],label=key)

plt.xlabel('Round No.')
plt.ylabel('Train Loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig("train_lenet_original")
plt.close()

start_pos = 0
r = [100*i for i in range(0,int(args['epochs']/100))]


for key in loss_train:
  # if(key!='Uncompressed'):
    plt.plot(r,test_acc_epochs[key][start_pos:],label=key)

plt.xlabel('Round No.')
plt.ylabel('Test Accuracy')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig("test_lenet_original")
