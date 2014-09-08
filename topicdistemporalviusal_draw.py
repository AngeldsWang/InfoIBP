'''
Created on 2014-5-29

@author: zhenjun.wang
'''
from __future__ import print_function
import pickle, os, sys
from LoadObservations import *
import matplotlib.pyplot as plt

time_slice_num = 20
start_year = 1993
user_num = 239
user_interval = (100, 200)
datalist = load_observations(time_slice_num, start_year, user_num)

with open("sample_likelihood0.3_1.plk", "rb") as picklefile:
    samples_likelihoods = pickle.load(picklefile)
    
picklefile.close()

samples = samples_likelihoods[0]
likelihoods = samples_likelihoods[1]

sample = samples[15]

user_source_mat = sample.user_source_mat
source_user_idx = sample.source_user_index
sourceuserid = 0
for source in source_user_idx:
    path = "source_" + str(source)
    if not os.path.exists(path):
        os.mkdir(path)
    for t in xrange(len(user_source_mat)):
        source_topids = datalist[0][t][source, :].tolist()
        fig_source_topicdis = plt.figure()
        user_infected_by_souce = np.nonzero(user_source_mat[t][:, sourceuserid])[0].tolist()
        
        plt.bar(range(len(source_topids)), source_topids)
        plt.ylim(0, 1)
        plt.savefig(path + "\\source_" + str(source) + "_topicdis_" + str(start_year + t) + ".png")
            
        usercount = 0
        for user in user_infected_by_souce:
            print("source user: ", source, end='')
            print(" in year ", start_year + t, end='')
            print(" infected user ", user)
            user_topicdis = datalist[0][t][user, :]
            fig_user_topicdis = plt.figure()
            plt.bar(range(len(user_topicdis)), user_topicdis)
            plt.ylim(0, 1)
            plt.savefig(path + "\\user_" + str(user) + "_topicdis_" + str(start_year + t) + ".png")
            if usercount > 5:
                break
            usercount += 1
                       
    sourceuserid += 1

            