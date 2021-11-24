import lenet

batch_size=128#Neural Network batch size
n_rnds=1500 #Global rounds
tau=5 #Local rounds
total_c=250 #Total no of clients

#no of clients to choose in each round. 'paper' toggles non-uniform client selection according to p_i's set by the data partition
no_of_c=[5] #['paper',10,40,60,80] 
local_epochs = 5
#p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
pi_min = 0.2

#brute force search for best lr
possible_lr= [0.1]#,0.07,0.05] 

plot_every_n = 20

K= 1
cluster =0

algo_lr = {  #lr for each algo
    0: 0.05,
    1:0.05,
    2:0.05
}

lrfactor = 0.8 #factor to reduce lr in scheduler

d_algo = {
            0: "MIFA",
            1: "UMIFA",
            2: "FedAvg"
        }