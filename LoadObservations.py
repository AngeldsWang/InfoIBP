import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import networkx as nx
import matplotlib.pyplot as plt

def load_observations(time_slice_num, start_year, user_num, user_mask=None):
    user_prefer_list = []
    user_adjacency_list = []

    if user_mask is None:
        user_mask = range(user_num)

    for t in xrange(time_slice_num):
        user_prefer_mat = np.loadtxt("userPrefer\\mostConnectedAuthorsTopicDis_" +
                                     str(time_slice_num) + "_" + str(start_year + t) + ".txt")
        user_prefer_list.append(user_prefer_mat[user_mask, :])

        user_adjacency_mat_sparse = np.loadtxt("userAdj\\mostConnectedCoauthors_" + str(time_slice_num) +
                                               "_" + str(start_year + t) + "_SPARSE.txt", dtype=int)

        assert user_adjacency_mat_sparse.shape[1] == 3

        row_idx = user_adjacency_mat_sparse[:, 0]
        col_idx = user_adjacency_mat_sparse[:, 1]
        value = user_adjacency_mat_sparse[:, 2]
        user_adjacency_mat = coo_matrix((value, (row_idx, col_idx)), shape=(user_num, user_num)).todense()
        user_adjacency_mat = np.array(user_adjacency_mat)
        user_adjacency_mat = (user_adjacency_mat + user_adjacency_mat.T > 0).astype(np.int)
        user_adjacency_list.append(user_adjacency_mat[np.ix_(user_mask, user_mask)])
    return user_prefer_list, user_adjacency_list

def load_DBLP_observations(time_slice_num, start_year, user_num, user_mask=None):
    user_prefer_list = []
    user_adjacency_list = []

    if user_mask is None:
        user_mask = range(user_num)

    for t in xrange(time_slice_num):
        user_prefer_mat = np.loadtxt("userTopicDistribution\\AuthorsTopicDis_" +
                                     str(time_slice_num) + "_" + str(start_year + t) + ".txt")
        user_prefer_list.append(user_prefer_mat[user_mask, :])

        user_adjacency_mat_sparse = np.loadtxt("userAdjacency\\Coauthors_" + str(time_slice_num) +
                                               "_" + str(start_year + t) + "_SPARSE.txt", dtype=int)

        assert user_adjacency_mat_sparse.shape[1] == 3

        row_idx = user_adjacency_mat_sparse[:, 0]
        col_idx = user_adjacency_mat_sparse[:, 1]
        value = user_adjacency_mat_sparse[:, 2]
        user_adjacency_mat = coo_matrix((value, (row_idx, col_idx)), shape=(user_num, user_num)).todense()
        user_adjacency_mat = np.array(user_adjacency_mat)
        user_adjacency_mat = (user_adjacency_mat + user_adjacency_mat.T > 0).astype(np.int)
        user_adjacency_list.append(user_adjacency_mat[np.ix_(user_mask, user_mask)])
    return user_prefer_list, user_adjacency_list


def load_digg_observations(time_slice_num, start_year, user_num, story_num, user_mask=None):
    user_prefer_list = []
    user_adjacency_list = []

    if user_mask is None:
        user_mask = range(user_num)
        
    active_user_ids = np.loadtxt("activeUsers.txt", dtype=np.int).tolist()
    remap_ids = range(len(active_user_ids))
    active_user_remap_ids = dict(zip(active_user_ids, remap_ids))
    

    for t in xrange(time_slice_num):
        user_prefer_mat_sparse = np.loadtxt("votings\\day_" + str(start_year + t) + "_voting.txt", dtype=int)
        user_idx = user_prefer_mat_sparse[:, 0]
        user_idx_remap = np.array([active_user_remap_ids[each] for each in user_idx])
        story_idx = user_prefer_mat_sparse[:, 1] - np.ones(user_prefer_mat_sparse[:, 1].shape)
        value = np.ones(user_prefer_mat_sparse[:, 1].shape)
        user_prefer_mat = csr_matrix((value, (user_idx_remap, story_idx)), shape=(user_num, story_num))
        user_prefer_list.append(user_prefer_mat[user_mask, :])

        user_adjacency_mat_sparse = np.loadtxt("friends\\day_" + str(start_year + t) + "_friends.txt", dtype=int)

        if user_adjacency_mat_sparse.shape[1] == 3:
            value = user_adjacency_mat_sparse[:, 2]
        else:
            value = np.ones((user_adjacency_mat_sparse[:, 0].shape), dtype=np.int)
        row_idx = user_adjacency_mat_sparse[:, 0].tolist()
        row_idx_remap = np.array([active_user_remap_ids[each] for each in row_idx])
        col_idx = user_adjacency_mat_sparse[:, 1].tolist()
        col_idx_remap = np.array([active_user_remap_ids[each] for each in col_idx])
        
        user_adjacency_mat = csr_matrix((value, (row_idx_remap, col_idx_remap)), shape=(user_num, user_num))
        user_adjacency_mat = (user_adjacency_mat + user_adjacency_mat.T > 0).astype(int)
        user_adjacency_list.append(user_adjacency_mat[np.ix_(user_mask, user_mask)])
    return user_prefer_list, user_adjacency_list


def find_most_connected_users(adj_mat):
    graph_adj = nx.from_numpy_matrix(adj_mat)
    sub_dense_graph_list = nx.connected_component_subgraphs(graph_adj)

    return sub_dense_graph_list[0].nodes()


if __name__ == "__main__":
    time_slice_num = 20
    start_year = 1993
    end_year = 2012
    user_num = 239
    (user_prefer, user_adjacency) = load_observations(time_slice_num, start_year, user_num)
    user_adj_0 = (user_adjacency[0] > 0).astype(np.int)
    dense_connected_users_idx = find_most_connected_users(user_adj_0)
    np.savetxt("more_most_connected_users_id.txt",
               np.array(dense_connected_users_idx).reshape(len(dense_connected_users_idx), 1), fmt="%d")