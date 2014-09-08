'''
Created on 2014-6-22

@author: zhenjun.wang
'''
import numpy as np
time_slice_num = 20
start_year = 1993
user_test_mask = np.arange(1300, 1400, 1).tolist()

for t in xrange(time_slice_num):
    user_adjacency_mat_sparse = np.loadtxt("userAdjacency\\Coauthors_" + str(time_slice_num) + 
                                           "_" + str(start_year + t) + "_SPARSE.txt", dtype=int)
    
    coauthor_pair = np.zeros((0, 2), dtype=np.int)

    for i in xrange(user_adjacency_mat_sparse.shape[0]):
        if (user_adjacency_mat_sparse[i, 0] not in user_test_mask) or (user_adjacency_mat_sparse[i, 1] not in user_test_mask):
            continue
        if user_adjacency_mat_sparse[i, 0] == user_adjacency_mat_sparse[i, 1]:
            continue
        
        coauthor_pair = np.vstack((coauthor_pair, user_adjacency_mat_sparse[i, 0:2]))

    np.savetxt("userAdjacency\\coauthor_1300_1400_" + str(t+start_year) + ".txt", coauthor_pair, fmt="%d")
        
    