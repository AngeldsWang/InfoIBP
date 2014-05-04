import numpy as np
import numpy.random as nR
import time
from IBPSampler import IBPSampler
import copy

np.set_printoptions(threshold=np.nan)

# parameters for the synthetic data
T = 10
U = 100
D = 20

num_chains = 5
samples_per_chain = 10
burn_in = 50
user_prefer_list = []
user_adj_list = []
for t in xrange(T):
    rd_mat = nR.rand(U, D)
    user_prefer_list.append(np.divide(rd_mat, np.tile(rd_mat.sum(axis=1).reshape(U, 1), (1, D))))
    rd_mat_again = nR.rand(U, U)
    user_adj_list.append((rd_mat_again > 0.95).astype(np.int))

datalist = [user_prefer_list, user_adj_list]

## init the user-infected user matrix with no more than `max_user_neighbor' neighbors at all time slice
#init_user_infected = [np.zeros((U, 1))] * T
#index_all = []
#assert len(init_user_infected) == len(user_adj_list)
#for t in xrange(T):
#    max_user_neighbors = 2
#    init_user_infect = np.zeros((U, U))
#    user_adj_t = user_adj_list[t]
#    for i in xrange(U):
#        idx = np.nonzero(user_adj_t[i, :])[0]
#        count = 0
#        while count < len(idx):
#            rand_pick_idx = np.floor(nR.uniform(0, 1) * len(idx))
#            init_user_infect[i, idx[rand_pick_idx]] = 1
#            count += 1
#            if count >= max_user_neighbors:
#                break
#
#    index = []
#
#    for j in xrange(U):
#        if np.count_nonzero(init_user_infect[:, j]) > 0:
#            index.append(j)
#            init_user_infected[t] = np.hstack((init_user_infected[t], init_user_infect[:, j].reshape(U, 1)))
#
#    init_user_infected[t] = np.delete(init_user_infected[t], 0, 1)
#    index_all.append(index)


# IBP parameter (gamma hyperparameters)
# for user infected matrix
(alpha_i, alpha_i_a, alpha_i_b) = (10., 10., 1.)
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
(gamma_i, delta_i) = (3., 1.)
# for markov transition prob parameters of infected user group matrix
(gamma_g, delta_g) = (1., 1.)

iter_num = 1
sample_gap = 1
current_sample = 1
samples = []
for chain in xrange(num_chains):

    nR.seed(int(time.time()))
    burn_in_count = 1

    print "start sampling the " + str(chain) + " chain.\n"

    IBPSer = IBPSampler(datalist, (alpha_i, alpha_i_a, alpha_i_b), (gamma_i, delta_i),
                        (alpha_g, alpha_g_a, alpha_g_b), (gamma_g, delta_g), (sigma_u, su_a, su_b),
                        (sigma_a, sa_a, sa_b), (sigma_w, sw_a, sw_b))   # , (init_user_infected, index_all))

    """ print and save for debug """
    print IBPSer.user_source_mat[0].shape
    print str(IBPSer.user_source_mat[0])
    print IBPSer.source_user_index
    np.savetxt("infmat_0", IBPSer.user_source_mat[0], "%d")
    np.savetxt("infmat_1", IBPSer.user_source_mat[1], "%d")
    print IBPSer.source_group_mat[0].shape
    print str(IBPSer.source_group_mat[0])
    print IBPSer.user_group_mat[0].shape
    print str(IBPSer.user_group_mat[0])
    print IBPSer.user_interact_mat.shape
    print str(IBPSer.user_interact_mat)
    np.savetxt("interactmat_0", IBPSer.user_interact_mat, "%.6f")

    while True:
        # sample an auxiliary slice variable s for all time slices
        s = nR.uniform(0, 1) * np.min(IBPSer.a_i)

        # extend some cols of zeros to user_source_mat for all time slices
        IBPSer.extend_representation(s, "i")

        """ print for debug """
        print IBPSer.user_source_mat[0].shape
        print str(IBPSer.user_source_mat[0])

        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()

        # sample the user_source_mat combining the two time series
        # the posterior P_{rs}^t = P(z^{(t-1)}=r|Y^{(t-1)}...Y^{(1)}, U^{(t-1)}...U^{(1)})*Q(r,s)*P(Y^{(t),U^{(t)}}|z^{(t)}=s)
        for i in xrange(IBPSer.user_num):
            for j in xrange(IBPSer.source_user_num):
                IBPSer.forward_backward_algorithm(i, j, "i")

        # delete columns from user_source_mat and recalculate the source_group_mat and user_group_mat
        # and change the parameters
        IBPSer.delete_empty_cols("i")

        print "current source user num is " + str(IBPSer.source_user_num) + "\n"

        # sample the alpha of user_source_mat
        IBPSer.sample_alpha("i")

        # sample the transition matrix prob parameters for non-empty columns of the user_source_mat
        IBPSer.sample_trans_prob_para_non_empty_cols("i")

        # make cache since some cols may be deleted
        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()

        # using Metropolis-Hastings samples the eps and user_interact_mat
        for i in xrange(5):
            IBPSer.sample_eps()

        IBPSer.cache_all_time_slice_calc_link = IBPSer.make_cache_calc_link()

        for i in xrange(10):
            IBPSer.sample_user_interact_mat()

        log_graph_likelihood = IBPSer.calc_graph_all_time_likelihood()
        print log_graph_likelihood

        print IBPSer.source_group_mat[0].shape
        print str(IBPSer.source_group_mat[0])
        # extend some cols of zeros to source_group_mat for all time slices
        s = nR.uniform(0, 1) * np.min(IBPSer.a_g)
        IBPSer.extend_representation(s, "g")

        print IBPSer.source_group_mat[0].shape
        print str(IBPSer.source_group_mat[0])

        # sample the source_group_mat
        # the posterior P_{rs}^t = P(g^{(t-1)}=r|U^{(t-1)}...U^{(1)})*Q(r,s)*P(U^{(t)}|g^{(t)}=s)
        for i in xrange(IBPSer.source_user_num):
            for j in xrange(IBPSer.active_group_num):
                IBPSer.forward_backward_algorithm(i, j, "g")

        print IBPSer.source_group_mat[0].shape
        print str(IBPSer.source_group_mat[0])

        # delete columns from source_group_mat and recalculate the user_group_mat
        # and change the parameters
        IBPSer.delete_empty_cols("g")

        print "current active group num is " + str(IBPSer.active_group_num) + "\n"

        # sample the alpha of source_group_mat
        IBPSer.sample_alpha("g")

        # sample the transition matrix prob parameters for non-empty columns of the source_group_mat
        IBPSer.sample_trans_prob_para_non_empty_cols("g")

        # sample group_prefer_mat
        if IBPSer.prefer_dis == "exp":
            IBPSer.sample_group_prefer_mat()
        if IBPSer.prefer_dis == "gaussian":
            IBPSer.group_prefer_posterior()

        log_prefer_likelihood = IBPSer.calc_user_prefer_likelihood()
        print log_prefer_likelihood

        if burn_in_count > burn_in and np.mod(iter_num - burn_in, sample_gap) == 0:
            samples.append(copy.deepcopy(IBPSer))
            if np.mod(current_sample, samples_per_chain) == 0:
                current_sample += 1
                iter_num += 1
                print "current_sample = " + str(current_sample)
                print "iter = " + str(iter_num)
                print "burn_in_count = " + str(burn_in_count)
                break
            current_sample += 1

        iter_num += 1
        burn_in_count += 1
        print "current_sample = " + str(current_sample)
        print "iter_num = " + str(iter_num)
        print "burn_in_count = " + str(burn_in_count)