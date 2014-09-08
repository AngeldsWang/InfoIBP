'''
Created on 2014-5-29

@author: zhenjun.wang
'''
from __future__ import print_function
import pickle, os, sys
from LoadObservations import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mt

time_slice_num = 20
start_year = 3
user_num = 393
story_num = 3553
user_interval = (100, 200)
user_mask = np.arange(0, 50, 1)
datalist = load_digg_observations(time_slice_num, start_year, user_num, story_num, user_mask)

with open("sample_likelihood0.3_digg.plk", "rb") as picklefile:
    samples_likelihoods = pickle.load(picklefile)
    
picklefile.close()

samples = samples_likelihoods[0]
likelihoods = samples_likelihoods[1]
sample_count = 0
for sample in samples:
    user_source_mat = sample.user_source_mat
    source_user_idx = sample.source_user_index
    print(source_user_idx)
    sourceuserid = 0
    
    for t in xrange(len(user_source_mat)):
        user_infected_by_all_souce = np.nonzero(np.sum(user_source_mat[t], 1))[0].tolist()
        usercount = 0
        for user in user_infected_by_all_souce:
            print("source user: ", source_user_idx, end='')
            print(" in year ", start_year + t, end='')
            print(" infected user ", user)
            usercount += 1
        print("totally influence: ", usercount, " users")
    
#     for source in source_user_idx:
#         path = "sample_" + str(sample_count) + "\\source_" + str(source)
#         if not os.path.exists(path):
#             os.makedirs(path)
#  
#         for t in xrange(0, len(user_source_mat)):
#              
# #             source_topids = datalist[0][t][source, :].tolist()
# #             fig_source_topicdis = plt.figure()
#             user_infected_by_souce = np.nonzero(user_source_mat[t][:, sourceuserid])[0].tolist()
#              
# #             plt.bar(range(len(source_topids)), source_topids)
# #             plt.ylim(0, 1)
# #             plt.savefig(path + "\\source_" + str(source) + "_topicdis_" + str(start_year + t) + ".png")
# #             plt.clf()
#                  
#             usercount = 0
#             for user in user_infected_by_souce:
#                 print("source user: ", source, end='')
#                 print(" in year ", start_year + t, end='')
#                 print(" infected user ", user)
# #                 user_topicdis = datalist[0][t][user, :]
# #                 fig_user_topicdis = plt.figure()
# #                 plt.bar(range(len(user_topicdis)), user_topicdis)
# #                 plt.ylim(0, 1)
# #                 plt.savefig(path + "\\user_" + str(user) + "_topicdis_" + str(start_year + t) + ".png")
# #                 plt.clf()
#                 usercount += 1
#             print("totally influence: ", usercount, " users")
#          
#         sourceuserid += 1
#     path1 = "sample_" + str(sample_count)
#     if not os.path.exists(path1):
#         os.makedirs(path1)
#     for t in xrange(len(user_source_mat)):
#         ax = plt.figure().gca()
#         user_source_mat_t = user_source_mat[t].astype(np.float)
#         ax.imshow(user_source_mat[t], interpolation="nearest", cmap=cm.gray, extent=[0,8,100,0])
#         ax.set_aspect(0.2)
#         ax.xaxis.set_major_locator(mt.NullLocator())
#         ax.yaxis.set_major_locator(mt.NullLocator())
#         
#         plt.savefig(path1 + "\\influence_propagation_time_" + str(t + 1) + ".png")
#         plt.clf()
    
    if sample_count == 0:
        break
    sample_count += 1

            