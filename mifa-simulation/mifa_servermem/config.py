batch_size=128 #Neural Network batch size
n_rnds= 300 #Global rounds
tau=5 #Local rounds
total_c=100 #Total no of clients

#no of clients to choose in each round. 'paper' toggles non-uniform client selection according to p_i's set by the data partition
no_of_c=[10] #['paper',10,40,60,80] 

#p_i min is for the paper's paradigm for client selection, each client is selected with a min prob of 0.2
pi_min = 0.2

#brute force search for best lr
possible_lr= [0.08,0.09,0.07,0.05] 

