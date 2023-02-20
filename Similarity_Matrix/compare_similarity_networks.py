# Copyright (C) 2021-2026, James Flamino


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
#http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import numpy as np
import networkx as nx
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from community import community_louvain
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.centrality import closeness_centrality
from networkx.algorithms.cuts import normalized_cut_size
import networkx.algorithms.community as nx_comm
import collections

from scipy import stats

import seaborn as sns

def random_partitition(G, k=2):
    partition = {}
    for node in G:
        partition[node] = np.random.randint(1, k + 1)
    return partition

def get_data(fname):
    data = pickle.load(open(fname, 'rb'))

    M = data['sim_matrix'] 
    nodes = data['nodes']
    tags = data['tags']

    edgelist = []
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i,j] > 0.0:
                edgelist.append((i, j, {'weight': M[i,j]}))

    G = nx.Graph(edgelist)

    return G, nodes

def get_data_fixed_nodes(fname, cap=500):
    data = pickle.load(open(fname, 'rb'))

    M = data['sim_matrix'] 
    nodes = data['nodes']
    tags = data['tags']

    if cap > len(M):
        target_nodes = np.arange(len(M))
    else:
        target_nodes = np.random.choice(len(M), cap, replace=False)

    edgelist = []
    for i in range(len(M)):
        for j in range(len(M)):
            if i != j and M[i,j] > 0.0:
                if i in target_nodes and j in target_nodes:
                    edgelist.append((i, j, {'weight': M[i,j]}))

    G = nx.Graph(edgelist)

    return G, nodes

def get_in_community_weights(G, partition):
    num_partitions = list(set(list(partition.values())))
    
    avg_cut = []
    node_dist = {}
    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)
        
        inner_weights = []
        for target_node in inner_nodes:
            for inner_node in inner_nodes:
                if inner_node != target_node and G.has_edge(target_node, inner_node):
                    inner_weights.append(G[target_node][inner_node]['weight'])


        print('Avg in-community weight for partition', partition_val, ':', np.mean(inner_weights))

def community_separation(G, nodes, partition):
    num_partitions = list(set(list(partition.values())))
    
    avg_cut = []
    for partition_val in num_partitions:
        inner_nodes = []
        outer_nodes = []
        for node, comm in partition.items():
            if comm == partition_val:
                inner_nodes.append(node)
            else:
                outer_nodes.append(node)
        avg_cut.append(normalized_cut_size(G, inner_nodes, outer_nodes, weight='weight'))
    return np.mean(avg_cut)

if __name__ == '__main__':
    mod_2020 = []
    mod_2016 = []

    cut_2020 = []
    cut_2016 = []

    steps = 10
    for i in range(steps):
        print(i + 1, '/', steps)
        print('Analyzing 2020 graph')
        G_2020, nodes_2020 = get_data_fixed_nodes('sim_network_quotes/sim_network_joined_large.pkl', cap=100) # 200, 500

        partition = community_louvain.best_partition(G_2020)

        target_mod_2020 = community_louvain.modularity(partition, G_2020)
        print('2020 modularity:', target_mod_2020)
        mod_2020.append(target_mod_2020)

        target_cut_2020 = community_separation(G_2020, nodes_2020, partition)
        print('2020 normalized cut:', target_cut_2020)
        cut_2020.append(target_cut_2020)

        # partition = random_partitition(G_2020, k=2)

        print('Analyzing 2016 graph')
        G_2016, nodes_2016 = get_data_fixed_nodes('../2016/sim_network_quotes/sim_network_joined_large.pkl', cap=100) # 200, 500
            
        partition = community_louvain.best_partition(G_2016)

        # get_in_community_weights(G_2016, partition)

        target_mod_2016 = community_louvain.modularity(partition, G_2016)
        print('2016 modularity:', target_mod_2016)
        mod_2016.append(target_mod_2016)

        target_cut_2016 = community_separation(G_2016, nodes_2016, partition)
        print('2016 normalized cut:', target_cut_2016)
        cut_2016.append(target_cut_2016)

        # partition = random_partitition(G_2016, k=2)

    print()
    print('Avg 2020 modulartiy:', np.mean(mod_2020))
    print('Avg 2016 modulartiy:', np.mean(mod_2016))
    print()
    print('Standard error 2020 modulartiy:', stats.sem(mod_2020, axis=None))
    print('Standard error 2016 modulartiy:', stats.sem(mod_2016, axis=None))
    print()
    print('Avg 2020 normalized cut:', np.mean(cut_2020))
    print('Avg 2016 normalized cut:', np.mean(cut_2016))
    print()
    print('Standard error 2020 normalized cut:', stats.sem(cut_2020, axis=None))
    print('Standard error 2016 normalized cut:', stats.sem(cut_2016, axis=None))