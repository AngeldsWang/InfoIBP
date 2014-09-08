'''
Created on 2014-6-10

@author: zhenjun.wang
'''

import pickle
from LoadObservations import load_observations
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mt
from matplotlib.ticker import MaxNLocator

def calc_user_activity(user_adj_list, user_source_mat, i, j, t):
    user_friends_idx = np.nonzero(user_adj_list[t][i, :])[0].tolist()
    friends_influence = 0
    if len(user_friends_idx) > 0:
        for friend in user_friends_idx:
            friends_influence += user_source_mat[t][friend, j]
        friends_influence /= len(user_friends_idx)
    
    all_influence = user_source_mat[t][i, j] + friends_influence
    user_i_activity = 1.0 / (1 + np.exp(-all_influence))
    
    return user_i_activity


def calc_actual_expected_infected_num(samples, user_adj_list, train_slice_num, K,  persistence=0.5):

    sample_count = 0
    
    p_susceptible_t_all_samples = np.zeros(train_slice_num)
    actual_infected_t_all_samples = np.zeros(train_slice_num)
    
    source_user_idx = samples[0].source_user_index
    
    for sample in samples:
        user_source_mat = sample.user_source_mat
        assert source_user_idx == sample.source_user_index
        
        susceptibility = sample.a_i
        assert len(susceptibility) == user_source_mat[0].shape[1]
        
    #     path1 = "sample_" + str(sample_count)
    #     if not os.path.exists(path1):
    #         os.makedirs(path1)
        for t in xrange(len(user_source_mat)):
            p_susceptible_t = 0
            for i in xrange(first_num):
                p_stubborn = 1
                for j in xrange(len(susceptibility)):
                    user_i_activity_t = calc_user_activity(user_adj_list, user_source_mat, i, j, t)
                    if t == 0:
                        p_susceptible_t_persistence = user_i_activity_t * susceptibility[j]
                    p_susceptible_t_persistence = persistence * user_source_mat[t - 1][i, j] + (1 - persistence) * user_i_activity_t * susceptibility[j]
                    p_stubborn *= (1 - p_susceptible_t_persistence)
                
                p_susceptible = 1 - p_stubborn
                
                p_susceptible_t += p_susceptible
                
            actual_infected_t = len(np.where(np.sum(user_source_mat[t], 1) > 0)[0])
            actual_infected_t_all_samples[t] += actual_infected_t
            p_susceptible_t_all_samples[t] += p_susceptible_t
    #         print "Influentials: " + str(source_user_idx) + " at timestep " + str(t + 1) + " influence " + str(p_susceptible_t) + " people in expectation."
    #         print "actually influece " + str(actual_infected_t) + " people.\n"
    #         ax = plt.figure().gca()
    #         user_source_mat_t = user_source_mat[t].astype(np.float)
    #         ax.imshow(user_source_mat[t], interpolation="nearest", cmap=cm.gray, extent=[0,8,100,0])
    #         ax.set_aspect(0.2)
    #         ax.xaxis.set_major_locator(mt.NullLocator())
    #         ax.yaxis.set_major_locator(mt.NullLocator())
    #         
    #         plt.savefig(path1 + "\\influence_propagation_time_" + str(t + 1) + ".png")
    #         plt.clf()
        
        
        sample_count += 1
        
    p_susceptible_t_all_samples /= len(samples)
    # set first time step infected user num is K (only themselves)
    p_susceptible_t_all_samples[0] = K
    actual_infected_t_all_samples /= len(samples)
    actual_infected_t_all_samples = actual_infected_t_all_samples.astype(np.int)
    
    for t in xrange(train_slice_num):
        print "Influentials: " + str(source_user_idx) + " at timestep " + str(t + 1) + " average influence " + str(p_susceptible_t_all_samples[t]) + " people."
        print "actually influece " + str(actual_infected_t_all_samples[t]) + " people.\n"
    
    return p_susceptible_t_all_samples, actual_infected_t_all_samples


    
if __name__=="__main__":
    
    time_slice_num = 20
    train_slice_num = 10
    start_year = 1993
    user_num = 239
    user_interval = (100, 200)
    first_num = 100
    persistence = 0.8
    datalist = load_observations(time_slice_num, start_year, user_num, range(first_num))
    
    with open("1_sample_likelihood_b_beta1_count_spread_0.3.plk", "rb") as picklefile:
        samples_likelihoods_1 = pickle.load(picklefile)
        
    picklefile.close()
    
    with open("2_sample_likelihood_beta1_count_spread0.3.plk", "rb") as picklefile:
        samples_likelihoods_2 = pickle.load(picklefile)
        
    picklefile.close()
    
    with open("3_sample_likelihood_beta10.3.plk", "rb") as picklefile:
        samples_likelihoods_3 = pickle.load(picklefile)
        
    picklefile.close()
    
    active_nodes_DD = np.loadtxt("active_nodes_DD_3.txt", dtype=np.int)
    active_nodes_GG = np.loadtxt("active_nodes_GG_3.txt", dtype=np.int)
    
    user_adj_list = datalist[1]
    samples1 = samples_likelihoods_1[0]
    samples2 = samples_likelihoods_2[0]
    samples3 = samples_likelihoods_3[0]
    
    expected_infected_1, actual_infected_1 = calc_actual_expected_infected_num(samples1, user_adj_list, train_slice_num, 1, persistence)
    expected_infected_2, actual_infected_2 = calc_actual_expected_infected_num(samples2, user_adj_list, train_slice_num, 2, persistence)
    expected_infected_3, actual_infected_3 = calc_actual_expected_infected_num(samples3, user_adj_list, train_slice_num, 3, persistence)
    
    plt.clf()
    ax = plt.subplot(111)
    plt.plot(range(train_slice_num), actual_infected_1, linewidth=3, linestyle='-', color='b', marker='o', label=r'$K = 1$')
    plt.plot(range(train_slice_num), expected_infected_1, linewidth=3, linestyle=':', color='b', marker='o', label=r'$\hat{K} = 1$')
    plt.plot(range(train_slice_num), actual_infected_2, linewidth=3, linestyle='-', color='g', marker='^', label=r'$K = 2$')
    plt.plot(range(train_slice_num), expected_infected_2, linewidth=3, linestyle=':', color='g', marker='^', label=r'$\hat{K} = 2$')
    plt.plot(range(train_slice_num), actual_infected_3, linewidth=3, linestyle='-', color='r', marker='s', label=r'$K = 3$')
    plt.plot(range(train_slice_num), expected_infected_3, linewidth=3, linestyle=':', color='r', marker='s', label=r'$\hat{K} = 3$')
    
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    plt.subplots_adjust(left=0.18, bottom=0.12, top=0.75)
    
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tick_params(axis='both', which='minor', labelsize=22)
    
    plt.xlabel("Time step", fontsize=22)
    plt.ylabel("Influence spread", fontsize=22)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), prop={'size': 22}, fancybox=True, shadow=True, ncol=3)
    plt.savefig("DBLP_influence_spread_actual_vs_estimation_" + str(persistence) + ".png")
    
    plt.clf()
    ax1 = plt.subplot(111)
    plt.plot(range(train_slice_num), active_nodes_DD[0, :], linewidth=3, linestyle='-', color='b', marker='o', label=r'$K = 1$')
    plt.plot(range(train_slice_num), expected_infected_1, linewidth=3, linestyle=':', color='b', marker='o', label=r'$\hat{K} = 1$')
    plt.plot(range(train_slice_num), active_nodes_DD[1, :], linewidth=3, linestyle='-', color='g', marker='^', label=r'$K = 2$')
    plt.plot(range(train_slice_num), expected_infected_2, linewidth=3, linestyle=':', color='g', marker='^', label=r'$\hat{K} = 2$')
    plt.plot(range(train_slice_num), active_nodes_DD[2, :], linewidth=3, linestyle='-', color='r', marker='s', label=r'$K = 3$')
    plt.plot(range(train_slice_num), expected_infected_3, linewidth=3, linestyle=':', color='r', marker='s', label=r'$\hat{K} = 3$')
    
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    plt.subplots_adjust(left=0.18, bottom=0.12, top=0.75)
    
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tick_params(axis='both', which='minor', labelsize=22)
    
    plt.xlabel("Time step", fontsize=22)
    plt.ylabel("Influence spread", fontsize=22)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), prop={'size': 22}, fancybox=True, shadow=True, ncol=3)
    plt.savefig("DBLP_influence_spread_DegreeDiscountIC_vs_estimation_" + str(persistence) + ".png")
    
    plt.clf()
    ax2 = plt.subplot(111)
    plt.plot(range(train_slice_num), active_nodes_GG[0, :], linewidth=3, linestyle='-', color='b', marker='o', label=r'$K = 1$')
    plt.plot(range(train_slice_num), expected_infected_1, linewidth=3, linestyle=':', color='b', marker='o', label=r'$\hat{K} = 1$')
    plt.plot(range(train_slice_num), active_nodes_GG[1, :], linewidth=3, linestyle='-', color='g', marker='^', label=r'$K = 2$')
    plt.plot(range(train_slice_num), expected_infected_2, linewidth=3, linestyle=':', color='g', marker='^', label=r'$\hat{K} = 2$')
    plt.plot(range(train_slice_num), active_nodes_GG[2, :], linewidth=3, linestyle='-', color='r', marker='s', label=r'$K = 3$')
    plt.plot(range(train_slice_num), expected_infected_3, linewidth=3, linestyle=':', color='r', marker='s', label=r'$\hat{K} = 3$')
    
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    plt.subplots_adjust(left=0.18, bottom=0.12, top=0.75)
    
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tick_params(axis='both', which='minor', labelsize=22)
    
    plt.xlabel("Time step", fontsize=22)
    plt.ylabel("Influence spread", fontsize=22)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), prop={'size': 22}, fancybox=True, shadow=True, ncol=3)
    plt.savefig("DBLP_influence_spread_GeneralGreedy_vs_estimation_" + str(persistence) + ".png")