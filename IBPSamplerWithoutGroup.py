import numpy as np
import numpy.random as nR
import scipy.stats as spST
import scipy.special as spSP
import numpy.linalg as nL
import copy

# We will be taking log(0) = -Inf, so turn off this warning
#np.seterr(divide='ignore')


class IBPSamplerWithoutGroup(object):
    """
    init the data and parameters
    datalist is a list of observational data
    the other arguments are the parameters of the model
    """

    def __init__(self, datalist, alpha_i, gamma_delta_i, sigma_u, sigma_a, sigma_w, budget):

        # load observations
        # user_prefer_list contain time slices of user preference matrix
        # user_adj_list contain time slices of user adjacency matrix
        self.user_prefer_list = datalist[0]
        self.user_adj_list = datalist[1]
        assert len(self.user_adj_list) == len(self.user_prefer_list)
        self.time_slice_num = len(self.user_adj_list)
        self.init_source_num = 10
        self.user_adj_is_symmetric = False
        self.budget  = budget
        self.infected_num_count = []

        (self.user_num, self.prefer_dim) = self.user_prefer_list[0].shape
        self.harmonic_num_n = np.sum(1.0 / np.array(range(1, self.user_num * self.time_slice_num + 1)))
        
        self.user_adj_degree = self.count_degree()

        # set parameters and hyperparameters
        if type(alpha_i) == tuple:
            (self.alpha_i, self.alpha_i_a, self.alpha_i_b) = alpha_i
        else:
            (self.alpha_i, self.alpha_i_a, self.alpha_i_b) = (alpha_i, None, None)

        if type(sigma_u) == tuple:
            (self.sigma_u, self.sigma_u_a, self.sigma_u_b) = sigma_u
        else:
            (self.sigma_u, self.sigma_u_a, self.sigma_u_b) = (sigma_u, None, None)

        if type(sigma_a) == tuple:
            (self.sigma_a, self.sigma_a_a, self.sigma_a_b) = sigma_a
        else:
            (self.sigma_a, self.sigma_a_a, self.sigma_a_b) = (sigma_a, None, None)

        if type(sigma_w) == tuple:
            (self.sigma_w, self.sigma_w_a, self.sigma_w_b) = sigma_w
        else:
            (self.sigma_w, self.sigma_w_a, self.sigma_w_b) = (sigma_w, None, None)

        if type(gamma_delta_i) == tuple:
            (self.gamma_i, self.delta_i) = gamma_delta_i
        else:
            print "input gamma_delta_i as a tuple"

        self.beta_sparsity = 1
        # init the eps for IWI'
        default_density = 0.1
        self.eps = np.log(default_density / (1 - default_density))

        # init the markov transition prob parameters
        self.a_i = [0] * self.init_source_num
        self.b_i = [0] * self.init_source_num

        self.trans_prob_para = {"i": (self.alpha_i, self.beta_sparsity, self.alpha_i_a, self.alpha_i_b,
                                      self.a_i, self.b_i, self.gamma_i, self.delta_i)}
        self.sticking_break_init("i")

        # init the latent variables
        (user_source_mat, (source_index, rest_index)) = self.init_inf_group_mat_by_mixture()
        self.user_source_mat = user_source_mat
        self.source_user_index = source_index
        self.rest_index = rest_index
        
        assert self.init_source_num == self.user_source_mat[0].shape[1]
        # init the mixture weight
        self.user_source_weight_mat = nR.normal(0, self.sigma_a, (self.user_num, self.init_source_num))

        self.trans_mat_list = {"i": self.user_source_mat}

        # init the influential preference matrix list
        self.influential_prefer_list = self.init_influential_prefer()

        # set the source users' number
        self.source_user_num = self.user_source_mat[0].shape[1]

        # init the user interact matrix
        self.is_user_interact_mat_identity = False
        if not self.is_user_interact_mat_identity:
            user_interact_mat = np.triu(nR.randn(self.source_user_num, self.source_user_num) * self.sigma_w, 0)
            user_interact_mat = user_interact_mat + user_interact_mat.T
            diag_idx = np.diag_indices(self.source_user_num)
            user_interact_mat[diag_idx] = 1
            self.user_interact_mat = user_interact_mat
        else:
            self.user_interact_mat = np.eye(self.source_user_num)

        # init the cache for calculating the prob of link
        self.cache_all_time_slice_calc_link = self.make_cache_calc_link()
        
    def count_degree(self):
        user_adj_degree = []
        for t in xrange(self.time_slice_num):
            user_adj = self.user_adj_list[t]
            user_adj_degree_t = np.sum(user_adj, 1)
        user_adj_degree.append(user_adj_degree_t)
        
        return user_adj_degree

    def init_inf_group_mat_by_mixture(self):
        user_source_mat_all = []
        user_source_mat = np.zeros((0, 0))

        # set beta parameter
        self.beta_sparsity = 1

        # for time 0
        for i in xrange(1, self.user_num + 1):
            # sample existing features (source users)
            user_source_mat_i = (nR.uniform(0, 1, (1, user_source_mat.shape[1])) <
                                 user_source_mat.sum(axis=0).astype(np.float) / (self.beta_sparsity + i - 1)).astype(np.int)

            # sample new features (source users)
            new_infected = spST.poisson.rvs(self.alpha_i * self.beta_sparsity / (self.beta_sparsity + i - 1))

            user_source_mat_i = np.hstack((user_source_mat_i, np.ones((1, new_infected))))

            # add to the user infected matrix
            user_source_mat = np.hstack((user_source_mat, np.zeros((user_source_mat.shape[0], new_infected))))
            user_source_mat = np.vstack((user_source_mat, user_source_mat_i))

        user_source_mat_all.append(user_source_mat)
        self.init_source_num = user_source_mat.shape[1]
        self.source_user_num = user_source_mat.shape[1]

        # use idx record the picked source user id
        idx = []
        # use users_left record the users id who will be picked
        users_left = range(self.user_num)
        for i in xrange(self.source_user_num):
            assert len(users_left) > 0
            userid = users_left[np.floor(nR.uniform(0, 1) * len(users_left)).astype(np.int)]
            idx.append(userid)
            users_left.remove(userid)

        # resample a b
        self.a_i = [0] * self.init_source_num
        self.b_i = [0] * self.init_source_num
        self.trans_prob_para = {"i": (self.alpha_i, self.beta_sparsity, self.alpha_i_a, self.alpha_i_b, self.a_i,
                                      self.b_i, self.gamma_i, self.delta_i)}
        self.sticking_break_init("i")

        for t in xrange(1, self.time_slice_num):
            user_source_mat_all.append(np.zeros((self.user_num, self.init_source_num)))

        # for the next T-1 time slice
        for t in xrange(1, self.time_slice_num):
            for i in xrange(self.user_num):
                # if we have add new source users, add them to groups
                for j in xrange(self.init_source_num):
                    if user_source_mat_all[t - 1][i, j] == 0:
                        user_will_be_infected = self.a_i[j]
                    else:
                        user_will_be_infected = self.b_i[j]

                    user_source_mat_all[t][i, j] = (nR.uniform(0, 1) < user_will_be_infected)

        return user_source_mat_all, (idx, users_left)

    # def init_group_prefer(self, distri='exp'):
    #     group_prefer_mat = []
    #     if distri == 'exp':
    #         theta = self.sigma_a
    #         for t in xrange(self.time_slice_num):
    #             group_num_t = self.source_group_mat[t].shape[1]
    #             assert group_num_t == self.user_group_mat[t].shape[1]
    #             rd_mat = nR.exponential(theta, (group_num_t, self.prefer_dim))
    #             group_prefer_mat.append(np.divide(rd_mat,
    #                                               np.tile(rd_mat.sum(axis=1).reshape(group_num_t, 1),
    #                                                       (1, self.prefer_dim))))
    #     if distri == 'gaussian':
    #         sigma = self.sigma_a
    #         for t in xrange(self.time_slice_num):
    #             group_num_t = self.source_group_mat[t].shape[1]
    #             assert group_num_t == self.user_group_mat[t].shape[1]
    #             rd_mat = nR.randn((group_num_t, self.prefer_dim)) * sigma
    #             group_prefer_mat.append(np.divide(rd_mat,
    #                                               np.tile(rd_mat.sum(axis=1).reshape(group_num_t, 1),
    #                                                       (1, self.prefer_dim))))
    #     return group_prefer_mat

    def init_influential_prefer(self):
        influential_prefer_list = []
        for t in xrange(self.time_slice_num):
            influential_prefer_mat = np.zeros((0, self.prefer_dim))
            for i in self.source_user_index:
                influential_prefer_mat = np.vstack((influential_prefer_mat, self.user_prefer_list[t][[i], :]))
            influential_prefer_list.append(influential_prefer_mat)

        return influential_prefer_list

    def sticking_break_init(self, obj_key):
        """
        sample the transition prob parameters by sticking-break process
        :param obj_key: the matrix name key
        """
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]
        assert len(a) == len(b)
        for k in xrange(len(a)):
            if k == 0:
                a[k] = nR.beta(alpha, beta_sparsity)
            else:
                stick_radio = nR.beta(alpha, 1)
                a[k] = a[k - 1] * stick_radio
            b[k] = nR.beta(gamma, delta)
        self.trans_prob_para[obj_key] = (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta)

    def extend_representation(self, s, obj_key):
        """
        using slice sampling to extend the empty columns of user_source_mat or source_group_mat
        :param s: slice variable
        :param obj_key: the key of current sampling matrix
        """
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]
        a_k_minusone = np.min(a)
        while a_k_minusone > s:
            x_prime = nR.rand() * a_k_minusone
            a_k = self.slice_sampling(1, 10, x_prime, 2, obj_key)
            if obj_key == "i":
                if a_k > s:
                    if len(self.rest_index) == 0:
                        return
                    if len(a) > self.budget:
                        return
                    
                    # count current infection
                    current_infected_num = []
                    for t in xrange(self.time_slice_num):
                        temp = np.sum(self.user_source_mat[t], axis=1)
                        current_infected_num.append(len(np.where(temp > 0)[0].tolist()))
                    self.infected_num_count.append((self.source_user_num, current_infected_num))
                    
                    a.append(a_k)
                    b.append(nR.beta(gamma, delta))
                    id_extend = self.rest_index[np.floor(nR.uniform(0, 1) * len(self.rest_index)).astype(np.int)]
                    self.source_user_index.append(id_extend)
                    self.rest_index.remove(id_extend)
                    for t in xrange(self.time_slice_num):
                        self.user_source_mat[t] = np.hstack((self.user_source_mat[t],
                                                             np.zeros((self.user_source_mat[t].shape[0], 1))))
                        self.influential_prefer_list[t] = np.vstack((self.influential_prefer_list[t],
                                                                     self.user_prefer_list[t][[id_extend], :]))
                        assert self.user_source_mat[t].shape[1] == len(a) == len(b) == self.influential_prefer_list[t].shape[0]
                        assert len(a) == len(self.source_user_index) == self.user_num - len(self.rest_index)

                    # rebind the trans_mat_dict
                    self.trans_mat_list["i"] = self.user_source_mat

                a_k_minusone = a_k

                #resize the user interact matrix
                num_new_zero_infect = len(a) - self.source_user_num
                if not self.is_user_interact_mat_identity:
                    if num_new_zero_infect == 0:
                        return
                    self.user_interact_mat = np.hstack((self.user_interact_mat,
                                                        nR.normal(0, self.sigma_w, (self.source_user_num, num_new_zero_infect))))
                    self.user_interact_mat = np.vstack((self.user_interact_mat,
                                                        nR.normal(0, self.sigma_w, (num_new_zero_infect, len(a)))))
                else:
                    self.user_interact_mat = np.eye(num_new_zero_infect + self.source_user_num)
                    
                #resize the user source prefer mixture matrix
                if num_new_zero_infect == 0:
                        return
                self.user_source_weight_mat = np.hstack((self.user_source_weight_mat,
                                                    nR.normal(0, self.sigma_a, (self.user_num, num_new_zero_infect))))

                self.cache_all_time_slice_calc_link = self.make_cache_calc_link()
                self.source_user_num = len(a)
                
        # count current infection
        current_infected_num = []
        for t in xrange(self.time_slice_num):
            temp = np.sum(self.user_source_mat[t], axis=1)
            current_infected_num.append(len(np.where(temp > 0)[0].tolist()))
        self.infected_num_count.append((self.source_user_num, current_infected_num))

        self.trans_prob_para[obj_key] = (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta)

    def slice_sampling(self, sample_num, burn, x_prime, step_size, obj_key):
        """
        Refer to the Iain Murray's matlab implementation and the pseudo code in David MacKay's text book p375
        :param sample_num: draw N samples from some distribution
        :param burn: the iteration number, discard the burn in loop period after reaching this number
        :param step_size: the step size of extend sampling interval
        :param x_prime: the initial sampled x
        :param obj_key: the key of current sampling matrix

        :return: the sampled x from some distribution P(x)
        """
        ## get the sample dim
        #if type(x_prime) == np.float:
        #    data_dim = 1
        #elif type(x_prime) == list or np.ndarray:
        #    data_dim = len(x_prime)
        #x_samples = np.zeros((sample_num, data_dim))
        #
        ## convert x to D-dim
        #if type(step_size) == np.int or np.float:
        #    step_size = np.array([step_size] * data_dim).reshape(1, data_dim)
        #elif type(step_size) == list and len(step_size) == 1:
        #    step_size = np.array(step_size * data_dim).reshape(1, data_dim)
        #elif type(step_size) == np.ndarray and step_size.shape == (1, 1):
        #    step_size = np.tile(step_size, (1, data_dim))
        #else:
        #    print "invalid step_size, please use numeric, list or numpy.ndarray"
        #    return
        #x_samples = [0] * sample_num
        log_prob = self.sticking_break_posterior(x_prime, obj_key)

        for i in xrange(sample_num + burn):
            log_prob_prime = np.log(nR.uniform(0, 1)) + log_prob
            ## sample the sampling dim
            #d = np.ceil(nR.uniform(0, 1) * data_dim) - 1
            #x_l = x_prime
            #x_r = x_prime
            x = x_prime
            ratio = nR.uniform(0, 1)
            x_l = x_prime - ratio * step_size
            x_r = x_prime + ratio * step_size

            while self.sticking_break_posterior(x_l, obj_key) > log_prob_prime:
                x_l = x_l - step_size

            while self.sticking_break_posterior(x_r, obj_key) > log_prob_prime:
                x_r = x_r + step_size

            # inner loop
            while True:
                x = nR.uniform(0, 1) * (x_r - x_l) + x_l
                log_prob = self.sticking_break_posterior(x, obj_key)
                if log_prob > log_prob_prime:
                    break
                else:
                    if x > x_prime:
                        x_r = x
                    else:
                        x_l = x

            x_prime = x

            if i >= burn:
                break

        return x_prime

    def sticking_break_posterior(self, x_k, obj_key):
        """
        calc the posterior prob of the length of breaking-left
        :param x_k: the length of breaking-left at the k-th breaking
        :param obj_key: the key of the current sampling matrix
        :rtype : type as x_k
        """
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]

        if x_k < 0 or x_k > np.min(a):
            return -np.inf

        effective_rows = 0

        if obj_key == "i":
            effective_rows = self.user_num * self.time_slice_num
        if obj_key == "g":
            effective_rows = self.source_user_num * self.time_slice_num

        log_posterior = (alpha - 1) * np.log(x_k) + effective_rows * np.log(1 - x_k)
        for i in xrange(1, effective_rows + 1):
            log_posterior += alpha * np.power((1 - x_k), i).astype(np.float) / i

        return log_posterior

    def make_cache_calc_link(self):
        cache_all_time_slice = []
        cache_one_time_slice_key_list = ["wi", "iwi", "siwi", "log_siwi", "log_one_minus_siwi"]
        for t in xrange(self.time_slice_num):
            wi = np.dot(self.user_interact_mat, self.user_source_mat[t].T).astype(np.float64)
            iwi = np.dot(self.user_source_mat[t], wi).astype(np.float64)
            siwi = 1 / (1 + np.exp(-iwi + self.eps)).astype(np.float64) - np.finfo(np.float64).eps
            log_siwi = np.log(siwi).astype(np.float64)
            log_one_minus_siwi = np.log(1 - siwi).astype(np.float64)
            cache_one_time_slice_value_list = [wi, iwi, siwi, log_siwi, log_one_minus_siwi]
            cache_one_time_slice_key_value = zip(cache_one_time_slice_key_list, cache_one_time_slice_value_list)
            cache_all_time_slice.append(dict(cache_one_time_slice_key_value))

        return cache_all_time_slice
    
    def calc_user_activity(self, i, j, t):
        user_friends_idx = np.nonzero(self.user_adj_list[t][i, :])[0].tolist()
        friends_influence = 0
        if len(user_friends_idx) > 0:
            for friend in user_friends_idx:
                friends_influence += self.user_source_mat[t][friend, j]
            friends_influence /= len(user_friends_idx)
        
        all_influence = self.user_source_mat[t][i, j] + friends_influence
        user_i_activity = 1.0 / (1 + np.exp(-all_influence))
        
        return user_i_activity
    
    def forward_backward_algorithm_with_relations(self, i, j, obj_key):
        """
        forward-backward sampling algorithm for an entry of the matrix
        :param i: the index of user i or source user i
        :param j: the index of source user j or group j
        :param obj_key: the key of current sampling matrix
        """
        # load parameters
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]
        trans_mat = self.trans_mat_list[obj_key]

        # prob_susceptible is the prob of i being infected by j
        # prob_stubborn is the prob of user_j not being infected by j
        user_i_activity_0 = self.calc_user_activity(i, j, 0)
        if user_i_activity_0 == 0:
            user_i_activity_0 = 0.5
        prob_susceptible = a[j]
        prob_stubborn = b[j]

        # build the transition prior matrix as log_q
        log_q = np.array([[np.log(1 - prob_susceptible), np.log(prob_susceptible)],
                          [np.log(1 - prob_stubborn), np.log(prob_stubborn)]])

        # init the user_source_mat variables' transition matrix for all time slices
        prob_trans_mat = []
        for t in xrange(self.time_slice_num):
            prob_trans_mat.append(np.zeros((2, 2)))

        # forward process
        # using the log_p_t_minus_one caches the (t-1)-th time slice transition prob p_{rs}^{t-1}
        # log_p_t_minus_one only has two state variables: 0 or 1
        # for time 0 we only need to calculate log_p_t_minus_one as the transitional prior plus data likelihood at time 0

        log_p_t_minus_one = np.zeros((2, 1))
        
        # if the infectious variable at time 0 was 0
        if trans_mat[0][i, j] == 0:
            # calc the posterior for state 0
            if obj_key == "i":
                log_p_t_minus_one[0] = np.log((1 - prob_susceptible * user_i_activity_0)) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
                

            # try to set the variable to the other state
            trans_mat[0][i, j] = 1

            self.update_one_oracle(i, j, 0, 1, obj_key)

            # calc the posterior for state 1
            if obj_key == "i":
                log_p_t_minus_one[1] = np.log(prob_susceptible * user_i_activity_0) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)

        # if the infectious variable at time 0 was 1 do the corresponding operations as above
        else:
            if obj_key == "i":
                log_p_t_minus_one[1] = np.log(prob_susceptible * user_i_activity_0) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[1] = np.log(prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

            trans_mat[0][i, j] = 0
            self.update_one_oracle(i, j, 0, 0, obj_key)

            if obj_key == "i":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible * user_i_activity_0) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

        # for the next time slices
        # we need to calculate posterior transition prob as the sum of three parts:
        # posterior at time t-1: log_p_t_minus_one
        # transitional prior: log_q
        # data likelihood: data likelihood at time t
        for t in xrange(1, self.time_slice_num):
            user_i_activity_t = self.calc_user_activity(i, j, t)
            if user_i_activity_t == 0:
                user_i_activity_t = 0.5
                
            log_q = np.array([[np.log(1 - prob_susceptible * user_i_activity_t), np.log(prob_susceptible * user_i_activity_t)],
                          [np.log(1 - prob_stubborn), np.log(prob_stubborn)]])
            
            # for every state at time t-1
            for r in [0, 1]:
                if trans_mat[t][i, j] == 0:
                    # try to set the state to be 0 at time t
                    s = 0
                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

                    s = 1
                    trans_mat[t][i, j] = s
                    self.update_one_oracle(i, j, t, s, obj_key)

                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s
                else:
                    s = 1
                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

                    s = 0
                    trans_mat[t][i, j] = s
                    self.update_one_oracle(i, j, t, s, obj_key)

                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

            # normalize the prob trans mat
            prob_trans_mat[t] = self.normalize_log_probs(prob_trans_mat[t])

            # check for avoiding numerical problem
            if np.count_nonzero(prob_trans_mat[t]) < 4:
                prob_trans_mat[t] += 0.0001
                prob_trans_mat[t] = prob_trans_mat[t] / np.sum(prob_trans_mat)

            # marginalize the posterior state prob at time t-1
            log_p_t_minus_one[0] = np.sum(prob_trans_mat[t][:, 0])
            log_p_t_minus_one[1] = np.sum(prob_trans_mat[t][:, 1])

            assert False == np.isinf(np.sum(log_p_t_minus_one))

        # backward process
        # for the last time slice
        # normalize the posterior prob of state at the last time slice
        prob_t_end = self.normalize_log_probs(log_p_t_minus_one)

        # cache the current infectious variable
        previous_usr_source_value = trans_mat[-1][i, j]

        # sample the posterior prob of the last infectious variable based on
        # the state posterior prob at the last time slice
        trans_mat[-1][i, j] = nR.rand() < prob_t_end[1]
        if trans_mat[-1][i, j] != previous_usr_source_value:
            self.update_one_oracle(i, j, -1, trans_mat[-1][i, j], obj_key)

        if self.time_slice_num == 1:
            return

        # backward sampling all previous time slices
        t = self.time_slice_num - 2
        while t >= 0:
            # get the prob of state 1 and 0 for time t from the trans posterior prob of time t+1
            pr_one = prob_trans_mat[t + 1][1, trans_mat[t + 1][i, j]]
            pr_zero = prob_trans_mat[t + 1][0, trans_mat[t + 1][i, j]]
            # normalization
            pr_one = pr_one / (pr_one + pr_zero)

            assert pr_one != 0

            # cache the infectious variable for time t
            previous_usr_source_value = trans_mat[t][i, j]
            # resample the infectious variable
            trans_mat[t][i, j] = nR.rand() < pr_one
            # update intermediate variables if the sampled value is different
            if trans_mat[t][i, j] != previous_usr_source_value:
                self.update_one_oracle(i, j, t, trans_mat[t][i, j], obj_key)
            t -= 1

    def forward_backward_algorithm(self, i, j, obj_key):
        """
        forward-backward sampling algorithm for an entry of the matrix
        :param i: the index of user i or source user i
        :param j: the index of source user j or group j
        :param obj_key: the key of current sampling matrix
        """
        # load parameters
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]
        trans_mat = self.trans_mat_list[obj_key]

        # prob_susceptible is the prob of i being infected by j
        # prob_stubborn is the prob of user_j not being infected by j
        prob_susceptible = a[j]
        prob_stubborn = b[j]

        # build the transition prior matrix as log_q
        log_q = np.array([[np.log(1 - prob_susceptible), np.log(prob_susceptible)],
                          [np.log(1 - prob_stubborn), np.log(prob_stubborn)]])

        # init the user_source_mat variables' transition matrix for all time slices
        prob_trans_mat = []
        for t in xrange(self.time_slice_num):
            prob_trans_mat.append(np.zeros((2, 2)))

        # forward process
        # using the log_p_t_minus_one caches the (t-1)-th time slice transition prob p_{rs}^{t-1}
        # log_p_t_minus_one only has two state variables: 0 or 1
        # for time 0 we only need to calculate log_p_t_minus_one as the transitional prior plus data likelihood at time 0

        log_p_t_minus_one = np.zeros((2, 1))

        # if the infectious variable at time 0 was 0
        if trans_mat[0][i, j] == 0:
            # calc the posterior for state 0
            if obj_key == "i":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

            # try to set the variable to the other state
            trans_mat[0][i, j] = 1

            self.update_one_oracle(i, j, 0, 1, obj_key)

            # calc the posterior for state 1
            if obj_key == "i":
                log_p_t_minus_one[1] = np.log(prob_susceptible) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[1] = np.log(prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

        # if the infectious variable at time 0 was 1 do the corresponding operations as above
        else:
            if obj_key == "i":
                log_p_t_minus_one[1] = np.log(prob_susceptible) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[1] = np.log(prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

            trans_mat[0][i, j] = 0
            self.update_one_oracle(i, j, 0, 0, obj_key)

            if obj_key == "i":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible) + self.calc_graph_slice_likelihood(0) + self.calc_user_prefer_slice_likelihood(0)
            if obj_key == "g":
                log_p_t_minus_one[0] = np.log(1 - prob_susceptible) + self.calc_user_prefer_slice_likelihood(0)

        # for the next time slices
        # we need to calculate posterior transition prob as the sum of three parts:
        # posterior at time t-1: log_p_t_minus_one
        # transitional prior: log_q
        # data likelihood: data likelihood at time t
        for t in xrange(1, self.time_slice_num):
            # for every state at time t-1
            for r in [0, 1]:
                if trans_mat[t][i, j] == 0:
                    # try to set the state to be 0 at time t
                    s = 0
                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

                    s = 1
                    trans_mat[t][i, j] = s
                    self.update_one_oracle(i, j, t, s, obj_key)

                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s
                else:
                    s = 1
                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

                    s = 0
                    trans_mat[t][i, j] = s
                    self.update_one_oracle(i, j, t, s, obj_key)

                    log_pu_s = self.calc_user_prefer_slice_likelihood(t)
                    if obj_key == "i":
                        log_p_s = self.calc_graph_slice_likelihood(t)
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_p_s + log_pu_s
                    if obj_key == "g":
                        prob_trans_mat[t][r, s] = log_p_t_minus_one[r] + log_q[r, s] + log_pu_s

            # normalize the prob trans mat
            prob_trans_mat[t] = self.normalize_log_probs(prob_trans_mat[t])

            # check for avoiding numerical problem
            if np.count_nonzero(prob_trans_mat[t]) < 4:
                prob_trans_mat[t] += 0.0001
                prob_trans_mat[t] = prob_trans_mat[t] / np.sum(prob_trans_mat)

            # marginalize the posterior state prob at time t-1
            log_p_t_minus_one[0] = np.sum(prob_trans_mat[t][:, 0])
            log_p_t_minus_one[1] = np.sum(prob_trans_mat[t][:, 1])

            assert False == np.isinf(np.sum(log_p_t_minus_one))

        # backward process
        # for the last time slice
        # normalize the posterior prob of state at the last time slice
        prob_t_end = self.normalize_log_probs(log_p_t_minus_one)

        # cache the current infectious variable
        previous_usr_source_value = trans_mat[-1][i, j]

        # sample the posterior prob of the last infectious variable based on
        # the state posterior prob at the last time slice
        trans_mat[-1][i, j] = nR.rand() < prob_t_end[1]
        if trans_mat[-1][i, j] != previous_usr_source_value:
            self.update_one_oracle(i, j, -1, trans_mat[-1][i, j], obj_key)

        if self.time_slice_num == 1:
            return

        # backward sampling all previous time slices
        t = self.time_slice_num - 2
        while t >= 0:
            # get the prob of state 1 and 0 for time t from the trans posterior prob of time t+1
            pr_one = prob_trans_mat[t + 1][1, trans_mat[t + 1][i, j]]
            pr_zero = prob_trans_mat[t + 1][0, trans_mat[t + 1][i, j]]
            # normalization
            pr_one = pr_one / (pr_one + pr_zero)

            assert pr_one != 0

            # cache the infectious variable for time t
            previous_usr_source_value = trans_mat[t][i, j]
            # resample the infectious variable
            trans_mat[t][i, j] = nR.rand() < pr_one
            # update intermediate variables if the sampled value is different
            if trans_mat[t][i, j] != previous_usr_source_value:
                self.update_one_oracle(i, j, t, trans_mat[t][i, j], obj_key)
            t -= 1

    def calc_graph_slice_likelihood(self, t):
        """
        calculating the likelihood of the user adjacency matrix given the user_source_mat at time slice t
        :param t: time slice
        """
        graph = self.user_adj_list[t]
        log_siwi = self.cache_all_time_slice_calc_link[t]["log_siwi"]
        log_one_minus_siwi = self.cache_all_time_slice_calc_link[t]["log_one_minus_siwi"]
        ll_matrix = np.multiply(graph, log_siwi) + np.multiply((1 - graph), log_one_minus_siwi)

        if self.user_adj_is_symmetric:
            ll = np.sum(np.triu(ll_matrix, 1))
        else:
            ll_matrix = np.multiply(ll_matrix, (1 - np.eye(ll_matrix.shape[0])))
            ll = np.sum(ll_matrix)

        return ll

    def calc_graph_all_time_likelihood(self):
        log_likelihood = 0
        for t in xrange(self.time_slice_num):
            log_likelihood += self.calc_graph_slice_likelihood(t)

        return log_likelihood

    def update_user_group_mat(self):
        pass

    def update_one_oracle(self, i, j, t, on_flag, obj_key):
        """
        update one oracle of the user_source_mat and the prob calculating cache
        :param i: the i-th user
        :param j: the j-th source user
        :param t: time slice
        :param on_flag: flag for on or off (1 or 0)
        :param: obj_key: the key of current calculating matrix
        """
        if obj_key == "i":
            if on_flag:
                self.cache_all_time_slice_calc_link[t]["wi"][:, i] = \
                    self.cache_all_time_slice_calc_link[t]["wi"][:, i] + self.user_interact_mat[:, j]
                self.cache_all_time_slice_calc_link[t]["iwi"][i, :] = \
                    self.cache_all_time_slice_calc_link[t]["iwi"][i, :] + self.cache_all_time_slice_calc_link[t]["wi"][j, :]
                self.cache_all_time_slice_calc_link[t]["iwi"][:, i] = self.cache_all_time_slice_calc_link[t]["iwi"][i, :].T
            else:
                self.cache_all_time_slice_calc_link[t]["wi"][:, i] = \
                    self.cache_all_time_slice_calc_link[t]["wi"][:, i] - self.user_interact_mat[:, j]
                self.cache_all_time_slice_calc_link[t]["iwi"][i, :] = \
                    self.cache_all_time_slice_calc_link[t]["iwi"][i, :] - self.cache_all_time_slice_calc_link[t]["wi"][j, :]
                self.cache_all_time_slice_calc_link[t]["iwi"][:, i] = self.cache_all_time_slice_calc_link[t]["iwi"][i, :].T

            self.cache_all_time_slice_calc_link[t]["iwi"][i, i] = 0

            # update other cache variables
            # siwi
            self.cache_all_time_slice_calc_link[t]["siwi"][i, :] = \
                1 / (1 + np.exp(-(self.cache_all_time_slice_calc_link[t]["iwi"][i, :] + self.eps)))
            self.cache_all_time_slice_calc_link[t]["siwi"][:, i] = \
                self.cache_all_time_slice_calc_link[t]["siwi"][i, :].T
            # log_siwi
            self.cache_all_time_slice_calc_link[t]["log_siwi"][i, :] = \
                np.log(self.cache_all_time_slice_calc_link[t]["siwi"][i, :])
            self.cache_all_time_slice_calc_link[t]["log_siwi"][:, i] = \
                self.cache_all_time_slice_calc_link[t]["log_siwi"][i, :].T
            # log_one_minus_siwi
            self.cache_all_time_slice_calc_link[t]["log_one_minus_siwi"][i, :] = \
                np.log(1 - self.cache_all_time_slice_calc_link[t]["siwi"][i, :])
            self.cache_all_time_slice_calc_link[t]["log_one_minus_siwi"][:, i] = \
                self.cache_all_time_slice_calc_link[t]["log_one_minus_siwi"][i, :].T

    @staticmethod
    def normalize_log_probs(log_mat):
        """
        normalize the log probabilities of the matrix entries
        :param log_mat: the log matrix
        :rtype : the normalized log prob matrix
        """
        a = np.max(log_mat)
        b = np.exp(log_mat - a)
        total = np.sum(b)
        pr = b / total

        return pr

    def calc_user_prefer_slice_likelihood(self, t):
        """
        calculate the likelihood of user preference matrix given latent parts
        latent parts: user_source_mat, source_prefer
        :param t: time slice
        """
        user_prefer_mat = self.user_prefer_list[t]
        user_source_mat = self.user_source_mat[t]
        source_prefer_mat = self.influential_prefer_list[t]
        assert user_source_mat.shape[1] == source_prefer_mat.shape[0]
        assert user_source_mat.shape == self.user_source_weight_mat.shape
        residual = user_prefer_mat - np.dot(np.multiply(user_source_mat, self.user_source_weight_mat), source_prefer_mat)
        log_likelihood = - user_source_mat.shape[0] * self.prefer_dim / 2 * np.log(2 * np.pi * self.sigma_u ** 2) \
                         - 1 / (2 * self.sigma_u ** 2) * np.trace(np.dot(residual.T, residual))

        assert False == np.isnan(log_likelihood)

        return log_likelihood

    def calc_user_prefer_likelihood(self):
        log_likelihood = 0
        for t in xrange(self.time_slice_num):
            log_likelihood += self.calc_user_prefer_slice_likelihood(t)

        return log_likelihood

    def delete_empty_cols(self, obj_key):
        # load parameters
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]

        i = 0
        if obj_key == "i":
            while True:
                if i >= self.source_user_num:
                    break
                if self.source_user_num == 1:
                    break
                the_col = np.ones((self.user_num, 0))
                for t in xrange(self.time_slice_num):
                    the_col = np.hstack((the_col, self.user_source_mat[t][:, i].reshape(self.user_num, 1)))

                if np.count_nonzero(the_col) == 0:
                    del a[i]
                    del b[i]
                    self.user_interact_mat = np.delete(self.user_interact_mat, i, 0)
                    self.user_interact_mat = np.delete(self.user_interact_mat, i, 1)
                    self.user_source_weight_mat = np.delete(self.user_source_weight_mat, i, 1)
                    self.rest_index.append(self.source_user_index[i])
                    del self.source_user_index[i]
                    for t in xrange(self.time_slice_num):
                        self.user_source_mat[t] = np.delete(self.user_source_mat[t], i, 1)
                        self.influential_prefer_list[t] = np.delete(self.influential_prefer_list[t], i, 0)
                    i -= 1
                    # rebind the trans_mat_dict
                    self.trans_mat_list["i"] = self.user_source_mat

                i += 1

            self.source_user_num = len(a)
            self.trans_prob_para[obj_key] = (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta)

    def sample_alpha(self, obj_key):
        # load parameters
        """
        sample alpha hyperparameter
        """
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]

        if obj_key == "i":
            source_num = self.source_user_num
            alpha = nR.gamma(source_num + alpha_a, 1.0 / (self.harmonic_num_n + alpha_b))
#             beta_sparsity = 10 * alpha
            self.alpha_i = alpha
#             self.beta_sparsity = beta_sparsity
            self.trans_prob_para[obj_key] = (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta)

    def sample_trans_prob_para_non_empty_cols(self, obj_key):
        """
        sample the transition matrix parameters a and b for the non-empty columns
        of user_source_mat or source_group_mat
        """
        (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta) = self.trans_prob_para[obj_key]
        if obj_key == "i":
            for i in xrange(self.source_user_num):
                (c_0_1, c_0_x, c_1_1, c_1_x) = self.count_stick_sufficient_stats(i, obj_key)
                if c_0_1 != 0:
                    a[i] = nR.beta(c_0_1, (1 + c_0_x - c_0_1) * self.beta_sparsity)
                else:
                    assert len(a) == 1
                    a_k_minusone = alpha
                    a[i] = self.slice_sampling(1, 10, nR.rand() * a_k_minusone, 2, obj_key)
                b[i] = nR.beta(gamma + c_1_1, delta + c_1_x - c_1_1)
            self.trans_prob_para[obj_key] = (alpha, beta_sparsity, alpha_a, alpha_b, a, b, gamma, delta)

    def count_stick_sufficient_stats(self, k, obj_key):
        trans_mat = self.trans_mat_list[obj_key]

        c_1_x = 0
        c_0_1 = 0
        c_1_1 = 0
        previous_is_zero = True

        if obj_key == "i":
            obj_num = self.user_num

        # since at time 0, every obj is beginning with 0
        c_0_x = obj_num

        for i in xrange(obj_num):
            for t in xrange(self.time_slice_num):
                if trans_mat[t][i, k]:
                    if previous_is_zero:
                        c_0_1 += 1
                    else:
                        c_1_1 += 1
                    if t < self.time_slice_num - 1:
                        c_1_x += 1
                    previous_is_zero = False
                else:
                    if t < self.time_slice_num - 1:
                        c_0_x += 1
                    previous_is_zero = True

            previous_is_zero = True

        return c_0_1, c_0_x, c_1_1, c_1_x

    def sample_eps(self):
        """
        using Metropolis-Hastings updates eps with a normal proposal centered on the
        current location
        """
        # calc current likelihood
        current_likelihood = self.calc_graph_all_time_likelihood()
        current_eps = self.eps
        # calc current prior using centered normal distribution
        current_prior = np.log(spST.norm.pdf(self.eps, 0, self.sigma_w))

        # sample from gaussian centered at current eps
        self.eps = nR.normal(self.eps, self.sigma_w)

        # make cache since change eps
        temp_cache = copy.deepcopy(self.cache_all_time_slice_calc_link)
        self.cache_all_time_slice_calc_link = self.make_cache_calc_link()

        # calc proposal likelihood
        proposal_likelihood = self.calc_graph_all_time_likelihood()

        # calc proposal prior
        proposal_prior = np.log(spST.norm.pdf(self.eps, 0, self.sigma_w))

        # calc the acceptance ratio
        acceptance_ratio = np.exp(proposal_likelihood + proposal_prior - current_likelihood - current_prior)

        if nR.rand() > acceptance_ratio:
            self.eps = current_eps
            self.cache_all_time_slice_calc_link = copy.deepcopy(temp_cache)
            
    def sample_user_source_weight_mat(self):
        """
        calculate the posterior weight: 
        Sigma = (sigma_u^-1*A[I,:]*A[I,:].T + sigma_a^-1*eye(|I|))^-1
        Mean = Sigma*sigma_u^-1*A[I,:]*X[n,:]
        where I is the nonzero index of influentials
        """
        for t in xrange(self.time_slice_num):
            for n in xrange(self.user_num):
                nonzero_idx = np.nonzero(self.user_source_mat[t][n, :])[0].tolist()
                if len(nonzero_idx) > 0:
                    post_sigma = nL.inv(1.0 / self.sigma_u * np.dot(self.influential_prefer_list[t][nonzero_idx, :], 
                                                              self.influential_prefer_list[t][nonzero_idx, :].T) + 
                                        1.0 / self.sigma_a * np.eye(len(nonzero_idx)))
                    post_mean = 1.0 / self.sigma_u * np.dot(np.dot(post_sigma, 
                                                                   self.influential_prefer_list[t][nonzero_idx, :]), 
                                                            self.user_prefer_list[t][n, :].T)
                    self.user_source_weight_mat[n, nonzero_idx] = post_mean.T

    def sample_user_interact_mat(self):
        """
        using Metropolis-Hastings updates user_interact_mat with a normal proposal centered on the
        current location
        """
        for i in xrange(self.source_user_num):
            for j in xrange(self.source_user_num):

                # calc current likelihood
                current_likelihood = self.calc_graph_all_time_likelihood()
                current_entry = self.user_interact_mat[i, j]
                # calc current prior using centered normal distribution
                current_prior = np.log(spST.norm.pdf(current_entry, 0, self.sigma_w))

                # sample from gaussian centered at current entry
                self.user_interact_mat[i, j] = nR.normal(current_entry, self.sigma_w)
                self.user_interact_mat[j, i] = self.user_interact_mat[i, j]

                # make cache since change user_interact_mat
                temp_cache = copy.deepcopy(self.cache_all_time_slice_calc_link)
                self.update_cache_with_entry(i, j, current_entry)

                # calc proposal likelihood
                proposal_likelihood = self.calc_graph_all_time_likelihood()

                # calc proposal prior
                proposal_prior = np.log(spST.norm.pdf(self.user_interact_mat[i, j], 0, self.sigma_w))

                # calc the acceptance ratio
                acceptance_ratio = np.exp(proposal_likelihood + proposal_prior - current_likelihood - current_prior)

                if nR.rand() > acceptance_ratio:
                    self.user_interact_mat[i, j] = current_entry
                    self.user_interact_mat[j, i] = current_entry
                    self.cache_all_time_slice_calc_link = copy.deepcopy(temp_cache)

    def update_cache_with_entry(self, i, j, old_entry):
        for t in xrange(self.time_slice_num):
            delta1 = (self.user_interact_mat[i, j] - old_entry) * (self.user_source_mat[t][:, j].reshape(1, self.user_num))
            if i != j:
                delta2 = (self.user_interact_mat[i, j] - old_entry) * (self.user_source_mat[t][:, i].reshape(1, self.user_num))
            self.cache_all_time_slice_calc_link[t]["wi"][i, :] = self.cache_all_time_slice_calc_link[t]["wi"][i, :] + delta1
            if i != j:
                self.cache_all_time_slice_calc_link[t]["wi"][j, :] = self.cache_all_time_slice_calc_link[t]["wi"][j, :] + delta2
                self.cache_all_time_slice_calc_link[t]["iwi"] = \
                    self.cache_all_time_slice_calc_link[t]["iwi"] + np.dot(self.user_source_mat[t][:, [i, j]], np.vstack((delta1, delta2)))
            else:
                self.cache_all_time_slice_calc_link[t]["iwi"] = \
                    self.cache_all_time_slice_calc_link[t]["iwi"] + np.dot(self.user_source_mat[t][:, [i]], delta1)
            self.cache_all_time_slice_calc_link[t]["siwi"] = \
                1 / (1 + np.exp(-(self.cache_all_time_slice_calc_link[t]["iwi"] + self.eps))) - np.finfo(np.float64).eps
            self.cache_all_time_slice_calc_link[t]["log_siwi"] = np.log(self.cache_all_time_slice_calc_link[t]["siwi"])
            self.cache_all_time_slice_calc_link[t]["log_one_minus_siwi"] = np.log(1 - self.cache_all_time_slice_calc_link[t]["siwi"])

    def update_user_prefer_time_series(self, i, new_idx):
        # for time 0
        new_influential_follow_sequence = np.zeros((self.user_num, 1))
        new_influential_follow_sequence[new_idx] = 1
        self.user_source_mat[0][:, [i]] = new_influential_follow_sequence
        self.influential_prefer_list[0][[i], :] = self.user_prefer_list[0][[new_idx], :]
        # for next T-1 time slices
        for t in xrange(1, self.time_slice_num):
            for k in xrange(self.user_num):
                if self.user_source_mat[t - 1][k, i] == 0:
                    user_will_be_infected = self.a_i[i]
                else:
                    user_will_be_infected = self.b_i[i]

                self.user_source_mat[t][k, i] = (nR.uniform(0, 1) < user_will_be_infected)

            self.influential_prefer_list[t][[i], :] = self.user_prefer_list[t][[new_idx], :]
            
    def rest_user_degree_hist(self):
        aggregate_adj = self.user_adj_list[0]
        for t in xrange(1, self.time_slice_num):
            aggregate_adj += self.user_adj_list[t]
        
        aggregate_adj = (aggregate_adj > 0).astype(np.int)
        rest_users_adj = aggregate_adj[np.ix_(self.rest_index, self.rest_index)]
        hist = np.sum(rest_users_adj, 1).astype(np.float) / np.max(np.sum(rest_users_adj, 1))
        hist_ = np.array([1.0 / (1 + np.exp(-v)) for v in hist])
        idx = np.argsort(hist_)
        hist_sorted = hist[idx]
        
        return hist_sorted, idx
    
    def find_possible_influentials(self):
        hist_sorted, idx = self.rest_user_degree_hist()
        index = 0
        for each in (nR.rand() < hist_sorted).astype(np.int).tolist():
            if each == 1:
                return idx[index:]              
            index += 1
            
    def sample_influential_exp(self):
        """
        Using Metropolis-Hastings algorithm to sample the rest users who will be influentials
        """
        if len(self.rest_index) <= 0:
            return
        # source_user_index = np.array(self.source_user_index)
        # rest_index = np.array(self.rest_index)
        
        rest_condidates_idx = self.find_possible_influentials()        
        self.rest_index = np.array(self.rest_index)
        candidates = self.rest_index[rest_condidates_idx]
        
        for i in xrange(len(self.source_user_index)):
            for j in xrange(len(candidates)):
                current_likelihood = self.calc_user_prefer_likelihood() + self.calc_graph_all_time_likelihood()
                current_influential_idx = self.source_user_index[i]

                # try to set current influential index to rest_index[j]
                self.source_user_index[i] = candidates[j]
                candidates[j] = current_influential_idx
                self.rest_index[rest_condidates_idx[j]] = current_influential_idx

                cache_user_prefer_time_series = copy.deepcopy((self.user_source_mat, self.influential_prefer_list))
                cache_link_predict_intermediate = copy.deepcopy(self.cache_all_time_slice_calc_link)
                self.update_user_prefer_time_series(i, self.source_user_index[i])
                self.cache_all_time_slice_calc_link = self.make_cache_calc_link()

                proposal_likelihood = self.calc_user_prefer_likelihood() + self.calc_graph_all_time_likelihood()

                acceptance_ratio = np.exp(proposal_likelihood - current_likelihood)

                if nR.rand() > acceptance_ratio:
                    self.rest_index[rest_condidates_idx[j]] = self.source_user_index[i]
                    candidates[j] = self.source_user_index[i]
                    self.source_user_index[i] = current_influential_idx
                    self.user_source_mat = copy.deepcopy(cache_user_prefer_time_series[0])
                    self.influential_prefer_list = copy.deepcopy(cache_user_prefer_time_series[1])
                    self.cache_all_time_slice_calc_link = copy.deepcopy(cache_link_predict_intermediate)
                    
        self.rest_index = self.rest_index.tolist()

    def sample_influential(self):
        """
        Using Metropolis-Hastings algorithm to sample the rest users who will be influentials
        """
        if len(self.rest_index) <= 0:
            return
        # source_user_index = np.array(self.source_user_index)
        # rest_index = np.array(self.rest_index)
        
#         rest_condidates_idx = self.find_possible_influentials()        
#         rest_index = np.array(self.rest_index)
        
        for i in xrange(len(self.source_user_index)):
            for j in xrange(len(self.rest_index)):
                current_likelihood = self.calc_user_prefer_likelihood() + self.calc_graph_all_time_likelihood()
                current_influential_idx = self.source_user_index[i]

                # try to set current influential index to rest_index[j]
                self.source_user_index[i] = self.rest_index[j]
                self.rest_index[j] = current_influential_idx

                cache_user_prefer_time_series = copy.deepcopy((self.user_source_mat, self.influential_prefer_list))
                cache_link_predict_intermediate = copy.deepcopy(self.cache_all_time_slice_calc_link)
                self.update_user_prefer_time_series(i, self.source_user_index[i])
                self.cache_all_time_slice_calc_link = self.make_cache_calc_link()

                proposal_likelihood = self.calc_user_prefer_likelihood() + self.calc_graph_all_time_likelihood()

                acceptance_ratio = np.exp(proposal_likelihood - current_likelihood)

                if nR.rand() > acceptance_ratio:
                    self.rest_index[j] = self.source_user_index[i]
                    self.source_user_index[i] = current_influential_idx
                    self.user_source_mat = copy.deepcopy(cache_user_prefer_time_series[0])
                    self.influential_prefer_list = copy.deepcopy(cache_user_prefer_time_series[1])
                    self.cache_all_time_slice_calc_link = copy.deepcopy(cache_link_predict_intermediate)


class IBPSamplerTester(object):
    def __init__(self):
        pass

    @staticmethod
    def log_likelihood_future(models_samples, samples_per_model_sample, future_adj_prefer, type_likelihood='adj'):
        assert len(future_adj_prefer[0]) == len(future_adj_prefer[1])
        future_time_steps = len(future_adj_prefer[0])
        log_likelihood_samples = []
        for i in xrange(len(models_samples)):
            for j in xrange(samples_per_model_sample):
                sample = models_samples[i]
                cache_sample_user_source_mat = sample.user_source_mat
                sample.user_source_mat = IBPSamplerTester.sample_new_source_users(sample, future_time_steps)
                assert sample.user_source_mat[0].shape[1] == sample.user_interact_mat.shape[0]
                if type_likelihood == 'adj':
                    log_likelihood_samples.append(IBPSamplerTester.log_likelihood_adj(future_adj_prefer[1], sample))
                if type_likelihood == 'prefer':
                    log_likelihood_samples.append(IBPSamplerTester.log_likelihood_prefer(future_adj_prefer[0], sample))
                sample.user_source_mat = cache_sample_user_source_mat
        ll = IBPSamplerTester.log_of_sum(log_likelihood_samples) - (np.log(len(models_samples)) + np.log(samples_per_model_sample))
        sigma = np.std(log_likelihood_samples)
        return ll, sigma

    @staticmethod
    def sample_new_source_users(sample, future_time_steps):
        num_user = sample.user_source_mat[0].shape[0]
        num_influential = sample.user_source_mat[0].shape[1]
        future_user_source_mat = []
        for t in xrange(future_time_steps):
            future_user_source_mat.append(np.zeros((num_user, num_influential), dtype=np.int))
            for i in xrange(num_user):
                for j in xrange(num_influential):
                    if (t == 0 and sample.user_source_mat[-1][i, j] == 0) or (t >= 1 and future_user_source_mat[t - 1][i, j] == 0):
                        prob_susceptible = sample.a_i[j]
                    else:
                        prob_susceptible = sample.b_i[j]

                    future_user_source_mat[t][i, j] = int(nR.uniform(0, 1) < prob_susceptible)

        return future_user_source_mat

    @staticmethod
    def log_likelihood_adj(future_adjs, sample):
        time_step_num = len(future_adjs)
        likelihood = 0
        num_users = sample.user_source_mat[0].shape[0]
        for t in xrange(time_step_num):
            for i in xrange(num_users):
                for j in xrange(i + 1, num_users):
                    iwi = np.dot(sample.user_source_mat[t][i, :],
                                 np.dot(sample.user_interact_mat, sample.user_source_mat[t][j, :].T))
                    if future_adjs[t][i, j] == 1:
                        likelihood += np.log(1.0 / (1 + np.exp(-iwi + sample.eps)))
                    else:
                        likelihood += np.log(1.0 - 1.0 / (1 + np.exp(-iwi + sample.eps)))

        return likelihood

    @staticmethod
    def log_likelihood_prefer(future_prefers, sample):
        time_step_num = len(future_prefers)
        log_likelihood = 0
        for t in xrange(time_step_num):
            user_prefer_future = future_prefers[t]
            source_prefer_mat = user_prefer_future[sample.source_user_index, :]
            assert sample.user_source_mat[t].shape[1] == source_prefer_mat.shape[0]
            residual = user_prefer_future - np.dot(np.multiply(sample.user_source_mat[t], 
                                                               sample.user_source_weight_mat), 
                                                   source_prefer_mat)
            log_likelihood += - sample.user_source_mat[t].shape[0] * user_prefer_future.shape[1] / 2 * np.log(2 * np.pi * sample.sigma_u ** 2) \
                              - 1 / (2 * sample.sigma_u ** 2) * np.trace(np.dot(residual.T, residual))

            assert False == np.isnan(log_likelihood)

        return log_likelihood

    @staticmethod
    def log_of_sum(log_vector):
        max_idx = np.argmax(log_vector)
        max_value = log_vector[max_idx]
        del log_vector[max_idx]
        l_sum = max_value
        a_ccum = 1
        for item in log_vector:
            a_ccum += np.exp(item - max_value)
        l_sum += np.log(a_ccum)

        return l_sum

    # def sample_group_prefer_mat(self):
    #     for t in xrange(self.time_slice_num):
    #         # get the non-zero col's index
    #         pos_k_idx = np.where(np.sum(self.user_group_mat[t], 0) > 0)[0]
    #
    #         assert len(pos_k_idx) != 0
    #
    #         # calculate intermediate matrices
    #         zz = np.dot(self.user_group_mat[t].T, self.user_group_mat[t])
    #         zx = np.dot(self.user_group_mat[t].T, self.user_prefer_list[t])
    #
    #         for idx in pos_k_idx:
    #             not_k = np.setdiff1d(pos_k_idx, np.array(idx))
    #             variance = self.sigma_u ** 2 / zz[idx, idx]
    #             if len(not_k) == 0:
    #                 mean = ((1.0 / self.sigma_a) * self.sigma_u ** 2 + zx[[idx], :]) / zz[idx, idx]
    #             else:
    #                 mean = ((1.0 / self.sigma_a) * self.sigma_u ** 2 + zx[[idx], :] -
    #                         np.dot(zz[[idx], not_k.reshape(1, len(not_k))], self.group_prefer_mat[t][not_k, :])) / zz[idx, idx]
    #
    #             self.group_prefer_mat[t][idx, :] = self.sample_from_rectified_gaussian(mean, variance * np.ones((1, self.prefer_dim)))

    # def sample_from_rectified_gaussian(self, m, v):
    #     """
    #     sample random variables from rectified gaussian distribution given the specific parameters
    #     :param m: the mean of posterior
    #     :param v: the variance of posterior
    #     """
    #     x = np.zeros(m.shape)
    #     y = nR.uniform(0, 1, m.shape)
    #     j = -m / np.sqrt(2 * v)
    #     k = j > 26
    #     x[k] = np.log(y[k]) / (m[k] / v[k])
    #     R = spSP.erfc(np.abs(j[np.logical_not(k)]))
    #     x[np.logical_not(k)] = np.multiply(self.my_erfcinv(np.multiply(y[np.logical_not(k)], R) - np.multiply((j[np.logical_not(k)] < 0),
    #                                                                                                        (2 * y[np.logical_not(k)] + R - 2))),
    #                                        np.sqrt(2 * v[np.logical_not(k)])) + m[np.logical_not(k)]
    #
    #     assert np.sum((np.isinf(x)).astype(np.int)) == 0
    #
    #     return x

    # @staticmethod
    # def my_erfcinv(y):
    #     """
    #     the `my' version fix the bug: erfcinv with small arguments loses precision
    #     """
    #     return -spSP.ndtri(0.5 * y) / np.sqrt(2)

    # @staticmethod
    # def calc_m(feature_load_mat, sigma_x, sigma_a):
    #     """ Calculate m = (feature_load_mat' * feature_load_mat - (sigma_x^2) / (sigma_a^2) * I)^-1 """
    #     feature_num = feature_load_mat.shape[1]
    #     return np.linalg.inv(np.dot(feature_load_mat.T, feature_load_mat) + (sigma_x ** 2)
    #                          / (sigma_a ** 2) * np.eye(feature_num))
    #
    # def group_prefer_mat_posterior(self, observation_mat, feature_load_mat):
    #     """
    #     Return E[A|X,Z]
    #     Mean/covar of posterior over group_prefer_mat
    #     """
    #     m = self.calc_m(feature_load_mat, self.sigma_u, self.sigma_a)
    #     mean_group_prefer_mat = np.dot(m, np.dot(feature_load_mat.T, observation_mat))
    #     covar_group_prefer_mat = self.sigma_u ** 2 * mean_group_prefer_mat
    #     return mean_group_prefer_mat, covar_group_prefer_mat
    #
    # def group_prefer_posterior(self):
    #     for t in xrange(self.time_slice_num):
    #         feature_load_mat = self.user_group_mat[t]
    #         observation_mat = self.user_prefer_list[t]
    #         self.group_prefer_mat[t] = self.group_prefer_mat_posterior(observation_mat, feature_load_mat)