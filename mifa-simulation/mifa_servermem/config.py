import lenet
import numpy as np

#Generic Hyp
batch_size= 64 #Neural Network batch size
n_rnds= 10 #Global rounds
tau=5 #Local rounds --> not in use since we use local epochs
total_c= 250 #Total no of clients
no_of_c=[5] #participating clients
local_epochs = 5 
plot_local_train_loss = True


#Cluster
K= 55        #try for 5,10,20. Number of clusters is predetermined for static
cluster = 1
clust_mech = 'dynamic' #'static 

#Model
model_type = 'lenet' #shakespeare_lstm' #'shakespeare_lstm'#'cnnmnist'#'cnnmnist' #'r'
dataset = 'cifar10'

#Plotting
plot_every_n = 1



#Learning rate
algo_lr = {  #lr for each algo, length has to be the same for all algos
    0: [1/np.power(10,2.5)],
    1: [1/np.power(10,2)],
    2: [1/np.power(10,2)],
    3: [1/np.power(10,2)]
    }

lrfactor = {
    0:1, #factor to reduce lr in scheduler
    1:1,
    2:1,
    3:1
    }

sch_freq = 200 #scheduler every n rounds



#select algos to run
d_algo = {
            #0: "MIFA",
            1: "UMIFA",
            #2: "FedAvg",
            #3: "scaffold"
        }




#MIFA paper paradigm variables
#p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
pi_min = 0.2
sel_client_variablepi = False #variable no of clients selected each round according to pi assigned
enforce_cp_r1 = False #enforce all client part in 1st round


