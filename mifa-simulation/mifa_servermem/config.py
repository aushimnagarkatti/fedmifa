import lenet
import numpy as np

batch_size=64#Neural Network batch size
n_rnds=1500 #Global rounds
tau=5 #Local rounds --> not in use since we use local epochs
total_c=250 #Total no of clients

#no of clients to choose in each round. 'paper' toggles non-uniform client selection according to p_i's set by the data partition
no_of_c=[5] #['paper',10,40,60,80] 
local_epochs = 1
#p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
pi_min = 0.2

#brute force search for best lr

plot_every_n = 20

K= 1
cluster =0

algo_lr = {  #lr for each algo, length has to be the same for all algos
    0: [1/np.power(10,0.5), 1/np.power(10,1.5), 1/np.power(10,2)],
    1: [1/np.power(10,0.5), 1/np.power(10,1.5), 1/np.power(10,2)],
    2: [1/np.power(10,0.5), 1/np.power(10,1.5), 1/np.power(10,2)]
    }

lrfactor = {
    0:1, #factor to reduce lr in scheduler
    1:1,
    2:1
    }

d_algo = {
            0: "MIFA",
            1: "UMIFA",
            2: "FedAvg"
        }