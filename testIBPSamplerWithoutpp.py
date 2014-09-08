# from __future__ import print_function
import time
import copy
# import logging
import numpy.random as nR
import pickle
import matplotlib.pyplot as plt
import pp

from IBPSamplerWithoutGroup import *
from LoadObservations import *
from utils import parse_args

options = parse_args()

# logging.basicConfig(level=logging.INFO, format='%(message)s')
# logger = logging.getLogger()
# logger.addHandler(logging.FileHandler(options.logfile, 'a'))
# print = logger.info
# np.set_printoptions(threshold=np.nan)
 
time_slice_num = 20
train_time_slice_num = 19
test_time_slice_num = 1
start_year = 1993
user_mask = range(0, 100)
user_num = 2213
user_train_mask = np.arange(1300, 1400, 1)
datalist = load_DBLP_observations(time_slice_num, start_year, user_num, user_mask)
print("load data ok!\n")
 
train_data_list = ([prefer for prefer in datalist[0][0: train_time_slice_num]],
                   [adj for adj in datalist[1][0: train_time_slice_num]])
 
test_data_list = ([prefer for prefer in datalist[0][train_time_slice_num:]],
                  [adj for adj in datalist[1][train_time_slice_num:]])
 
print(str(datalist[1][0]))
 
# # parameters for the synthetic data
# T = 10
# U = 100
# D = 20
#
# num_chains = 5
# samples_per_chain = 10
# burn_in = 50
# user_prefer_list = []
# user_adj_list = []
# for t in xrange(T):
#     rd_mat = nR.rand(U, D)
#     user_prefer_list.append(np.divide(rd_mat, np.tile(rd_mat.sum(axis=1).reshape(U, 1), (1, D))))
#     rd_mat_again = nR.rand(U, U)
#     user_adj_list.append((rd_mat_again > 0.95).astype(np.int))
#
# datalist = [user_prefer_list, user_adj_list]
 
# IBP parameter (gamma hyperparameters)u
# for user infected matrix
 
(alpha_i, alpha_i_a, alpha_i_b) = (0.3, 1., 1.)
# for infected user group matrix
(alpha_g, alpha_g_a, alpha_g_b) = (1., 1., 1.)
# Observed data Gaussian noise (Gamma hyperparameters)
# for user preference matrix
(sigma_u, su_a, su_b) = (1., 1., 1.)
# for group preference matrix
(sigma_a, sa_a, sa_b) = (1., 1., 1.)
# for infected user correlated matrix
(sigma_w, sw_a, sw_b) = (.1, 1., 1.)
# for markov transition prob parameters of user infected matrix
(gamma_i, delta_i) = (1., 1.)
# for markov transition prob parameters of infected user group matrix
(gamma_g, delta_g) = (1., 1.)
   
samples = []
likelihood_all = {}
num_chains = 1
burn_in = 300
iter_num = 1
samples_per_chain = 10
sample_gap = 1
current_sample = 1
budget = 100
   
for chain in xrange(num_chains):
    nR.seed(int(time.time()))
    burn_in_count = 1
   
    print("start sampling the " + str(chain) + " chain.\n")
   
    IBPSer = IBPSamplerWithoutGroup(train_data_list, (alpha_i, alpha_i_a, alpha_i_b), (gamma_i, delta_i),
                                    (sigma_u, su_a, su_b), (sigma_a, sa_a, sa_b), (sigma_w, sw_a, sw_b), budget)
   
    """ print and save for debug """
    print(IBPSer.user_source_mat[0].shape)
    print(str(IBPSer.user_source_mat[0]))
    print(IBPSer.source_user_index)
    np.savetxt("user_source_0", IBPSer.user_source_mat[0], "%d")
    np.savetxt("user_source_1", IBPSer.user_source_mat[1], "%d")
    print(IBPSer.user_interact_mat.shape)
    print(str(IBPSer.user_interact_mat))
    np.savetxt("interact_mat_0", IBPSer.user_interact_mat, "%.6f")
   
    adj_likelihood_train = []
    prefer_likelihood_train = []
   
    while True:
        if len(IBPSer.a_i) == 0:
            break
        # sample an auxiliary slice variable s for all time slices
        s = nR.uniform(0, 1) * np.min(IBPSer.a_i)
   
        # extend some cols of zeros to user_source_mat for all time slices
        IBPSer.extend_representation(s, "i")
   
        """ print for debug """
        print("sampled before hmm\n")
        print(str(IBPSer.source_user_index))
        print(IBPSer.user_source_mat[-1].shape)
        print(str(IBPSer.user_source_mat[-1]))
   
        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()
   
        # sample the user_source_mat combining the two time series
        # the posterior P_{rs}^t = P(z^{(t-1)}=r|Y^{(t-1)}...Y^{(1)}, U^{(t-1)}...U^{(1)})*Q(r,s)*P(Y^{(t),U^{(t)}}|z^{(t)}=s)
        for i in xrange(IBPSer.user_num):
            for j in xrange(IBPSer.source_user_num):
                IBPSer.forward_backward_algorithm_with_relations(i, j, "i")
   
        print("sampled after hmm\n")
        print(IBPSer.user_source_mat[-1].shape)
        print(str(IBPSer.user_source_mat[-1]))
   
        # delete columns from user_source_mat and recalculate the source_group_mat and user_group_mat
        # and change the parameters
        IBPSer.delete_empty_cols("i")
   
        print("current source user num is " + str(IBPSer.source_user_num) + "\n")
   
        # sample the alpha of user_source_mat
        IBPSer.sample_alpha("i")
   
        # sample the transition matrix prob parameters for non-empty columns of the user_source_mat
        IBPSer.sample_trans_prob_para_non_empty_cols("i")
   
        # make cache since some cols may be deleted
        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()
   
        # using Metropolis-Hastings samples the eps and user_interact_mat
        IBPSer.sample_eps()
   
        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()
   
        IBPSer.sample_user_interact_mat()
           
        IBPSer.sample_user_source_weight_mat()
   
        log_graph_likelihood = IBPSer.calc_graph_all_time_likelihood()
        print("graph log likelihood: ", log_graph_likelihood)
        adj_likelihood_train.append(log_graph_likelihood)
   
        # sample influential_prefer_mat
        print("a_i = " + str(IBPSer.a_i))
        print("b_i = " + str(IBPSer.b_i))
        print(str(IBPSer.source_user_index))
        print("sampled before resampling influentials")
        print(str(IBPSer.user_source_mat[-1]))
        IBPSer.sample_influential()
        print(str(IBPSer.source_user_index))
        print("sampled after resampling influentials")
        print(str(IBPSer.user_source_mat[-1]))
   
        log_prefer_likelihood = IBPSer.calc_user_prefer_likelihood()
        print("prefer log likelihood:, ", log_prefer_likelihood)
        prefer_likelihood_train.append(log_prefer_likelihood)
   
        if burn_in_count > burn_in and np.mod(iter_num - burn_in, sample_gap) == 0:
            samples.append(copy.deepcopy(IBPSer))
            if np.mod(current_sample, samples_per_chain) == 0:
                current_sample += 1
                iter_num += 1
                print("current_sample = " + str(current_sample))
                print("iter = " + str(iter_num))
                print("burn_in_count = " + str(burn_in_count))
                assert len(prefer_likelihood_train) == len(adj_likelihood_train)
                likelihood_all[chain] = (prefer_likelihood_train, adj_likelihood_train)
                fig = plt.figure()
                x = range(1, len(prefer_likelihood_train) + 1)
                l1, l2 = plt.plot(x, prefer_likelihood_train, 'rs-', x, adj_likelihood_train, 'yd-')
                fig.legend((l1, l2), ('preference likelihood', 'adjacency likelihood'), 'center right')
                plt.savefig("likelihood_" + str(chain))
                break
            current_sample += 1
   
        iter_num += 1
        burn_in_count += 1
        print("current_sample = " + str(current_sample))
        print("iter_num = " + str(iter_num))
        print("burn_in_count = " + str(burn_in_count))
   
   
with open('sample_likelihood_19_1_1300_1400_source_beta1_count_spread_' + str(alpha_i) + '.plk', 'wb') as samples_likelihood_plk:
    pickle.dump((samples, likelihood_all), samples_likelihood_plk)
   
samples_likelihood_plk.close()
 
with open('sample_likelihood_19_1_1300_1400_source_beta1_count_spread_' + str(alpha_i) + '.plk', 'rb') as samples_likelihood_plk:
    samples = pickle.load(samples_likelihood_plk)
         
samples_likelihood_plk.close()
     
print "load samples ok!"
 
if len(samples[0]) > 0:
    adj_likelihood_all = []
    adj_sigma_all = []
    prefer_likelihood_all = []
    prefer_sigma_all = []
    for n in xrange(len(test_data_list[0])):    
        test_prefer_adj = ([test_data_list[0][i] for i in range(n + 1)], [test_data_list[1][i] for i in range(n + 1)])
        adj_likelihood, adj_sigma = IBPSamplerTester.log_likelihood_future(samples[0], 10, test_prefer_adj, 'adj')
        print("adj_likelihood = " + str(adj_likelihood))
        adj_likelihood_all.append(adj_likelihood)
        adj_sigma_all.append(adj_sigma)
        prefer_likelihood, prefer_sigma = IBPSamplerTester.log_likelihood_future(samples[0], 10, test_prefer_adj, 'prefer')
        print("prefer_likelihood = " + str(prefer_likelihood))
        prefer_likelihood_all.append(prefer_likelihood)
        prefer_sigma_all.append(prefer_sigma)
         
    with open('predict_likelihood_19_1300_1400_source_beta1_count_spread_' + str(alpha_i) + '.plk', 'wb') as predict_likelihood_plk:
        pickle.dump((adj_likelihood_all, adj_sigma_all, prefer_likelihood_all, prefer_sigma_all), predict_likelihood_plk)
 
    predict_likelihood_plk.close()
        
    fig_predict_likelihood = plt.figure()
    ax = plt.subplot(111)
    x = range(1, len(test_data_list[0]) + 1)
    eb1 = plt.errorbar(x, adj_likelihood_all, adj_sigma_all, fmt="o", label="adjacency likelihood")
    eb2 = plt.errorbar(x, prefer_likelihood_all, prefer_sigma_all, fmt="rs", label="preference likelihood")
    ax.legend(loc=3)
    plt.savefig("prediction_1300_1400_" + str(len(x)))

# with open("predict_likelihood0.3.plk", "rb") as predict_likelihood_plk:
#     predict_ = pickle.load(predict_likelihood_plk)
#     
# predict_likelihood_plk.close()
# 
# fig_predict_likelihood = plt.figure()
# ax = plt.subplot(111)
# x = range(1, len(predict_[0]) + 1)
# eb1 = plt.errorbar(x, predict_[0], predict_[1], fmt="o", label="adjacency likelihood")
# eb2 = plt.errorbar(x, predict_[2], predict_[3], fmt="rs", label="preference likelihood")
# ax.legend(loc=3)
# plt.savefig("prediction_" + str(len(x)))
    
    
    
        
        
    