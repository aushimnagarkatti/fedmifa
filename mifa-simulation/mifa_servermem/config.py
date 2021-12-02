import lenet
import numpy as np

#Generic Hyp
batch_size= 64 #Neural Network batch size
n_rnds= 10 #Global rounds
tau=5 #Local rounds --> not in use since we use local epochs
total_c= 250 #Total no of clients
no_of_c=[5] #participating clients
local_epochs = 5 


#Cluster
K= 5
cluster = 1

#Model
model_type = 'r'

#Plotting
plot_every_n = 20



#Learning rate
algo_lr = {  #lr for each algo, length has to be the same for all algos
    0: [1/np.power(10,2)],
    1: [1/np.power(10,2)],
    2: [1/np.power(10,2)]
    }

lrfactor = {
    0:0.9, #factor to reduce lr in scheduler
    1:0.9,
    2:0.9
    }

sch_freq = 200 #scheduler every n rounds



#select algos to run
d_algo = {
            0: "MIFA",
            1: "UMIFA",
            2: "FedAvg"
        }




#MIFA paper paradigm variables
#p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
pi_min = 0.2
sel_client_variablepi = False #variable no of clients selected each round according to pi assigned
enforce_cp_r1 = False #enforce all client part in 1st round



#try with scheduler
#umifa and mifa and all with scheduler
#mifa paper simulation using variable learning rate or without client participation in rd 1

#first ran variable clients+running all clients first round












# batch_size= 100 #Neural Network batch size
# n_rnds= 200 #Global rounds
# tau=5 #Local rounds --> not in use since we use local epochs
# total_c= 100 #Total no of clients
# no_of_c=[5] #participating clients
# local_epochs = 2 


# #Plotting
# plot_every_n = 1



# #Learning rate
# algo_lr = {  #lr for each algo, length has to be the same for all algos
#     0: [0.1],
#     1: [0.1],
#     2: [0.1]
#     }

# lrfactor = {
#     0:0.8, #factor to reduce lr in scheduler
#     1:0.8,
#     2:0.8
#     }

# sch_freq = 5 #scheduler every n rounds



# # #Cluster
# K= 5
# cluster = 0


# #select algos to run
# d_algo = {
#             0: "MIFA",
#            # 1: "UMIFA",
#             2: "FedAvg"
#         }

# model_type = 'l'


# #MIFA paper paradigm variables
# #p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
# pi_min = 0.2
# sel_client_variablepi = True #variable no of clients selected each round according to pi assigned
# enforce_cp_r1 = True #enforce all client part in 1st round