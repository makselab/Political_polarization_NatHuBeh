# Copyright (C) 2021-2026, Brendan Cross


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

import networkit as nk
import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import numpy as np
import pandas as pd
import pdb
import graph_tool.all as gt
import time
import operator
import pickle
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.spatial.distance import correlation
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from math import pi, sin, cos, radians

WORKING_DIR = '/home/crossb/working'
NUM_WORKERS = 32
THREADS_PER_WORKER = 2
Election_2020_dir = '/home/pub/hernan/Election_2020/joined_output'

BIAS_TO_RETWEET_NETWORKS = {
    'Center news': os.path.join(WORKING_DIR, '', 'Center news_retweet_edges.csv'),
    'Fake news': 'Fake news_retweet_edges.csv',
    'Extreme bias left': 'Extreme bias left_retweet_edges.csv',
    'Extreme bias right': 'Extreme bias right_retweet_edges.csv',
    'Left leaning news': 'Left leaning news_retweet_edges.csv',
    'Right leaning news': 'Right leaning news_retweet_edges.csv',
    'Left news': 'Left news_retweet_edges.csv',
    'Right news': 'Right news_retweet_edges.csv'
}
BIAS_TO_COLOR = {
    'center': 'mediumseagreen',
    'fake': 'saddlebrown',
    'left extreme': 'tab:pink',#'deeppink',#'hotpink',
    'left leaning': 'mediumblue',#'royalblue',
    'left': 'darkmagenta',#'purple',
    'right leaning': 'darkgreen',
    'right': 'orange',
    'right extreme': 'tab:red'
}
BIAS_TO_INT = {
    'center': 0,
    'fake': 1,
    'left extreme': 2,
    'left leaning': 3,
    'left': 4,
    'right leaning': 5,
    'right': 6,
    'right extreme': 7
}
BIAS_ENUM = {
    0: 'center',
    1: 'fake',
    2: 'left extreme',
    3: 'left leaning',
    4: 'left',
    5: 'right leaning',
    6: 'right',
    7: 'right extreme'
}

RT_GRAPHS_DIR_2016 = '/home/crossb/packaged_ci/graphs/2016/'
PATH_TO_BIAS_2016 = {
    os.path.join(RT_GRAPHS_DIR_2016, 'center_2016_ci.gt'): 'center',
    os.path.join(RT_GRAPHS_DIR_2016, 'extreme_bias_left_2016_ci.gt'): 'left extreme',
    os.path.join(RT_GRAPHS_DIR_2016, 'extreme_bias_right_2016_ci.gt'): 'right extreme',
    os.path.join(RT_GRAPHS_DIR_2016, 'fake_2016_ci.gt'): 'fake',
    os.path.join(RT_GRAPHS_DIR_2016, 'lean_left_2016_ci.gt'): 'left leaning',
    os.path.join(RT_GRAPHS_DIR_2016, 'lean_right_2016_ci.gt'): 'right leaning',
    os.path.join(RT_GRAPHS_DIR_2016, 'left_2016_ci.gt'): 'left',
    os.path.join(RT_GRAPHS_DIR_2016, 'right_2016_ci.gt'): 'right'
}


def initialize_local_cluster():
    """
    To parallelize our dask operations, intialize a local cluster and client.
    :return:
    """
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=THREADS_PER_WORKER,
                           scheduler_port=0, dashboard_address=None)
    client = Client(cluster)
    return client


def load_graphs_gt(path, year='2020', completeness='simple', x=-3):
    #pdb.set_trace()
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    graphs = {}
    remove_string = 'complete' if completeness == 'simple' else 'simple'
    for file in fnames:
        if 'pro_' in file or remove_string in file or 'combined' in file:
            continue
        if year == '2020':
            #bias = ' '.join(file.split('_')[:-2])
            bias = ' '.join(file.split('_')[:x]) # temporarily, since we have both complete and simple in one dir,
            # we will name the bias the full path so that we can get them all in one table
        elif year == '2016':
            bias = PATH_TO_BIAS_2016[os.path.join(path, file)]
        else:
            raise ValueError

        print('gt bias:', bias)
        graph = gt.load_graph(os.path.join(path, file))
        graphs[bias] = graph
    return graphs


def load_from_graphtool_nk(path):
    graph = nk.readGraph(path, nk.Format.GraphToolBinary)
    return graph


def top_influencers(N, graphs):
    """
    Gets the top N influencers of each graph in graphs dict
    :param num_influencers:
    :param graphs:
        A dictionary where key is the bias of the graph and the value is the graph-tool graph.
        *note: each vertex of the graph-tool graphs given has a CI property that was precalculated,
        giving us the vertex's rank.
    :return:
    """
    print("Get top influencers")
    stime = time.time()
    top_influencers_by_bias = {}
    for bias, graph in graphs.items():
        res = []
        for vertex in graph.vertices():
            res.append((graph.vp.CI_in[vertex], graph.vp.CI_out[vertex], graph.vp.user_id[vertex], vertex))
        res = sorted(res, key=lambda x: x[1], reverse=True)

        #top_influencers_by_bias[bias] = {rank+1: (x[3], x[2], x[1]) for rank, x in enumerate(res[:N])}
        top_influencers_by_bias[bias] = {x[2]: (rank+1, x[3], x[1]) for rank, x in enumerate(res[:N])}
    print("Gathering top influencers took: {} seconds".format(time.time() - stime))
    return top_influencers_by_bias


def top_influencer_network(bias_networks, influencers, keep_all_edges=False):
    """
    This method creates a single influencer network from the top influencer nodes / edges from
    each bias network.
    :param classified_edges:
    :param influencers:
    :return:
    """
    print("Create top influencer network")
    stime = time.time()
    leanings = sorted(list(influencers.keys()))

    influencer_edges = []
    user_to_bias = {uid: bias for bias, uids in influencers.items() for rank, (vertex, uid, ci) in uids.items()}

    all_influencers = set(list(user_to_bias.keys()))
    total_edges = 0
    accepted_edges = 0
    for leaning in leanings:
        # grab all edges associated with the influencers of this category
        #influencer_uids = influencers[leaning]
        graph = bias_networks[leaning]
        for source, target, tweet_id in graph.iter_edges([graph.ep.tweet_id]):
            s_uid = graph.vp.user_id[source]
            t_uid = graph.vp.user_id[target]

            if keep_all_edges or (s_uid in all_influencers and t_uid in all_influencers):
                influencer_edges.append((s_uid, t_uid, tweet_id, leaning, BIAS_TO_COLOR[leaning]))
                accepted_edges += 1
            total_edges += 1
    print("Total Edges considered:", total_edges)
    print("Edges accepted:", accepted_edges)

    print("Num influencer edges:", len(influencer_edges))
    influencer_network = gt.Graph(directed=True)
    influencer_network.vertex_properties['user_id'] = influencer_network.new_vertex_property('int64_t')
    influencer_network.edge_properties['tweet_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['source_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['bias'] = influencer_network.new_edge_property('string')

    influencer_network.edge_properties['color'] = influencer_network.new_edge_property('string')
    influencer_network.vp.user_id = influencer_network.add_edge_list(influencer_edges, hashed=True,
                                     eprops=[influencer_network.ep.tweet_id,
                                             influencer_network.ep.bias,
                                             influencer_network.ep.color])#,
                                             #influencer_network.ep.source_id])
    influencer_network.edge_properties['bias_e'] = influencer_network.new_edge_property('int64_t')
    for e in influencer_network.edges():
        influencer_network.ep.bias_e[e] = BIAS_TO_INT[influencer_network.ep.bias[e]]

    uid_to_new_vertex = {influencer_network.vp.user_id[v]: v for v in influencer_network.vertices()}

    # add a bias vertex property
    influencer_network.vertex_properties['pie_fractions'] = influencer_network.new_vertex_property('vector<int64_t>')
    influencer_network.vertex_properties['shape'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['text'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['text_color'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['size'] = influencer_network.new_vertex_property('double')
    influencer_network.vertex_properties['pie_colors'] = influencer_network.new_vertex_property('vector<string>')

    # calculate the highest rank each vertex got
    vertex_ranks = {}
    vertex_to_bias_ci = {}
    for bias, uids in influencers.items():
        for rank, (old_v, uid, ci) in uids.items():
            if uid not in uid_to_new_vertex.keys():
                continue
            vertex = uid_to_new_vertex[uid]

            if vertex not in vertex_to_bias_ci or vertex not in vertex_ranks:
                vertex_ranks[vertex] = {bias: 0 for bias in leanings}
                vertex_to_bias_ci[vertex] = {bias: ci}
            else:
                vertex_to_bias_ci[vertex][bias] = ci

            vertex_ranks[vertex][bias] = rank

    if not keep_all_edges:
        largest_ci_by_bias = {bias: max([ci for rank, (vertex, uid, ci) in uids.items()]) for bias, uids in influencers.items()}

        for v in influencer_network.vertices():
            (bias, highest_rank) = max([x for x in vertex_ranks[v].items() if x[1] != 0], key=operator.itemgetter(1))

            if highest_rank <= 5:
                influencer_network.vp.text[v] = '{}'.format(highest_rank)
            else:
                influencer_network.vp.text[v] = ''
            influencer_network.vp.text_color[v] = BIAS_TO_COLOR[bias]
            influencer_network.vp.shape[v] = 'pie'

            influencer_network.vp.pie_colors[v] = [BIAS_TO_COLOR[bias] for bias in leanings]
            influencer_network.vp.pie_fractions[v] = [vertex_ranks[v][bias] for bias in leanings]

            influencer_network.vp.size[v] = 7 + (vertex_to_bias_ci[v][bias] / largest_ci_by_bias[bias] * 13)

            ranks = np.array([1/x if x != 0 else 0 for x in influencer_network.vp.pie_fractions[v]])
            fractions = ranks/np.sum(ranks) * 100
            influencer_network.vp.pie_fractions[v] = list(fractions.astype(np.int64))

    print("Creating top influencer network took: {} seconds".format(time.time() - stime))
    return influencer_network


def hub_analysis(graph, node_to_uid):
    """
    Find the hubs in our given network and their neighborhood.
    :param graph:
        networkit graph
    :param node_to_uid:
        networkit graph node number to twitter user id
    :return:
    """
    # TODO: Networkit might have a method to find neighbors within some distance.
    # get hubs
    NUM_HUBS = 100
    node_degrees = np.array([[node, graph.degreeOut(node)] for node in node_to_uid.keys()])

    # top 20
    hubs = node_degrees[node_degrees[:,1].argsort()][::-1][:NUM_HUBS]
    hubs_uid = [(node_to_uid[node], degree) for (node, degree) in hubs]
    print("Hubs: {}".format(hubs))

    #hub_nbrs = {node: [x for x in graph.iterNeighbors(node)] for node, _ in hubs}
    #print("Hub neighbors")
    #for node, nbrs in hub_nbrs.items():
    #    print("{} neighbors: {}".format(node, nbrs))

    return hubs_uid


def influencer_network_anlaysis(year=2020):
    path_2016 = '/home/crossb/packaged_ci/graphs/2016/'
    path_2020 = '/home/crossb/packaged_ci/graphs/2020/'
    top_n_influencers = 30
    biased_graphs = load_graphs_gt(path_2020)
    biased_influencers = top_influencers(top_n_influencers, biased_graphs)
    influencer_network = top_influencer_network(biased_graphs, biased_influencers)
    stats = network_characteristics_gt(influencer_network)
    print("Influencer network stats")
    for stat, value in stats.items():
        print("{}: {}".format(stat, value))

    most_infl_influencers = top_influencers(10, {'top': influencer_network})
    print("Most influential:", most_infl_influencers)


    # save influencer network
    gt.save('data/2020/influencer_network.gt')

    # networkit stats
    nk_graph = load_from_graphtool_nk('data/2020/influencer_network.gt')
    characteristics = network_characteristics_nk(nk_graph, 10)

    for stat, value in characteristics.items():
        if "centrality" in stat:
            print("{}: {}".format(
                stat,
                ','.join(['(Node: {}: {})'.format(influencer_network.vp.user_id[n], v) for (n, v) in value])))
        else:
            print("{}: {}".format(stat, value))

    # Draw with the vertices as pie charts
    vprops = {'pie_fractions': influencer_network.vp.pie_fractions,
              'shape': influencer_network.vp.shape,
              'text': influencer_network.vp.text,
              'text_color': influencer_network.vp.text_color,
              'size': influencer_network.vp.size,
              'pie_colors': influencer_network.vp.pie_colors,
              'text_position': 200,
              'font_size': 14,
              'text_offset': [0.0, 1.0]
              }
    eprops = {'color': 'lightgray'}
    pos = gt.arf_layout(influencer_network, d=4, max_iter=0)
    gt.graph_draw(influencer_network, pos=pos, vprops=vprops,
                  eprops=eprops, output='top_influencers.svg')

    return


def network_characteristics_nk(graph, N):
    print("Gathering network statistics!")
    characteristics = {
        'n_nodes': graph.numberOfNodes(),
        'n_edges': graph.numberOfEdges(),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0,
        'in_heterogeneity': 0,
        'out_heterogeneity': 0,
        'eigenvector_centrality': [0] * N,
        'degree_centrality': [0] * N,
        'betweenness_centrality': [0] * N
    }

    # get in-degree distribution
    in_degrees = np.array([graph.degreeIn(node) for node in graph.iterNodes()])
    # get out-degree distribution
    out_degrees = np.array([graph.degreeOut(node) for node in graph.iterNodes()])

    characteristics['avg_degree'] = np.average([graph.degree(node) for node in graph.iterNodes()])
    characteristics['max_in_degree'] = np.max(in_degrees)
    characteristics['max_out_degree'] = np.max(out_degrees)

    characteristics['in_heterogeneity'] = np.std(in_degrees) / characteristics['avg_degree']
    characteristics['out_heterogeneity'] = np.std(out_degrees) / characteristics['avg_degree']

    # centrality measures
    # eigenvector centrality
    cent = nk.centrality.EigenvectorCentrality(graph)
    stime = time.time()
    cent.run()
    characteristics['eigenvector_centrality'] = cent.ranking()[:N]
    print("Eigenvector Centrality time taken: {} seconds".format(time.time() - stime))

    # Betweenness centrality
    cent = nk.centrality.ApproxBetweenness(graph, epsilon=0.1)
    stime = time.time()
    cent.run()
    characteristics['betweenness_centrality'] = cent.ranking()[:N]
    print("Betweenness Centrality time taken: {} seconds".format(time.time() - stime))

    # Degree centrality
    cent = nk.centrality.DegreeCentrality(graph)
    stime = time.time()
    cent.run()
    characteristics['degree_centrality'] = cent.ranking()[:N]
    print("Degree Centrality time taken: {} seconds".format(time.time() - stime))

    return characteristics


def network_characteristics_gt(graph, sample_size=10000):
    n_samples = 1000
    print("Gathering network statistics!")
    characteristics = {
        'n_nodes': graph.num_vertices(),
        'n_edges': graph.num_edges(),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0,
        'in_heterogeneity': 0,
        'out_heterogeneity': 0,
        'unique_tweets': 0,
        'unique_users': 0
    }
    nodes = graph.get_vertices()
    # get in-degree distribution
    in_degrees = np.array(graph.get_in_degrees(nodes))
    # get out-degree distribution
    out_degrees = np.array(graph.get_out_degrees(nodes))

    characteristics['avg_degree'] = np.average(graph.get_total_degrees(nodes))/2
    characteristics['max_in_degree'] = np.max(in_degrees)
    characteristics['max_out_degree'] = np.max(out_degrees)

    unique_tweets = set()
    for e in graph.edges():
        unique_tweets.add(graph.ep.tweet_id[e])
    characteristics['unique_tweets'] = len(unique_tweets)

    unique_users = set()
    for v in graph.vertices():
        unique_users.add(graph.vp.user_id[v])
    characteristics['unique_users'] = len(unique_users)

    # Calculate Heterogeneity
    in_degree_stdevs = []
    out_degree_stdevs = []
    in_std_error = []
    out_std_error = []
    in_avg_degree = []
    out_avg_degree = []
    het_in = []
    het_out = []
    for i in range(n_samples):
        in_samples = np.random.choice(in_degrees, sample_size, replace=True)
        out_samples = np.random.choice(out_degrees, sample_size, replace=True)
        in_degree_stdevs.append(np.std(in_samples))
        out_degree_stdevs.append(np.std(out_samples))
        in_std_error.append(sem(in_samples))
        out_std_error.append(sem(out_samples))
        in_avg_degree.append(np.average(in_samples))
        out_avg_degree.append(np.average(out_samples))
        het_in.append(np.std(in_samples) / np.average(in_samples))
        het_out.append(np.std(out_samples) / np.average(out_samples))

    o_in_error = np.std(het_in)
    o_out_error = np.std(het_out)

    characteristics['in_heterogeneity'] = '{} \pm {}'.format(np.average(het_in), o_in_error)
    characteristics['out_heterogeneity'] = '{} \pm {}'.format(np.average(het_out), o_out_error)
    return characteristics


def influence_gain(graphs_2016, graphs_2020, N=100, plot=False):
    influencers_by_bias_2016 = top_influencers(None, graphs_2016)
    influencers_by_bias_2020 = top_influencers(None, graphs_2020)

    user_to_ci_2016 = defaultdict(int)
    user_to_ci_2020 = defaultdict(int)

    user_to_bias_ci = {2016: defaultdict(dict), }

    bias_user_rankings = {2016: defaultdict(dict), 2020: defaultdict(dict)}

    # load user map pickle
    user_map = pickle.load(open('/home/crossb/packaged_ci/maps/user_map_2020.pkl', 'rb')) #2020
    apply_user_map = lambda x: user_map[int(x)]['name']
    verified_map = lambda x: user_map[x]['verified']
    user_stats = []

    name_to_id = {data['name']: uid for uid, data in user_map.items()}

    pdb.set_trace()

    for bias in graphs_2016.keys():
        graph_2016 = graphs_2016[bias]
        graph_2020 = graphs_2020[bias]

        bias_stats = []
        bias_2016 = defaultdict(dict)
        bias_2020 = defaultdict(dict)

        if bias == 'fake':
            pdb.set_trace()


        for v in graph_2016.vertices():
            uid = graph_2016.vp.user_id[v]
            if uid == 109065990:
                pdb.set_trace()
            user_to_ci_2016[uid] = user_to_ci_2016.get(uid, 0) + graph_2016.vp.CI_out[v]
            #user_to_ci_2016[uid][bias] = graph_2016.vp.CI_out[v]
            #bias_2016[uid] = graph_2016.vp.CI_out[v]
            bias_2016[uid]['CI_out'] = graph_2016.vp.CI_out[v]
            bias_2016[uid]['rank'] = influencers_by_bias_2016[bias][uid][0]
            #uid_to_vertex_2016[uid] = v
            bias_user_rankings[2016][uid].update({bias: graph_2016.vp.CI_out[v]})

        if bias == 'fake':
            pdb.set_trace()

        for v in graph_2020.vertices():
            uid = graph_2020.vp.user_id[v]
            user_to_ci_2020[uid] = user_to_ci_2020.get(uid, 0) + graph_2020.vp.CI_out[v]
            #user_to_ci_2020[uid][bias] = graph_2020.vp.CI_out[v]
            #bias_2020[uid] = graph_2020.vp.CI_out[v]
            bias_2020[uid]['CI_out'] = graph_2020.vp.CI_out[v]
            bias_2020[uid]['rank'] = influencers_by_bias_2020[bias][uid][0]
            bias_user_rankings[2020][uid].update({bias: graph_2020.vp.CI_out[v]})

        for uid in set(bias_2016.keys()) | set(bias_2020.keys()):
            ci_2016 = bias_2016[uid].get('CI_out', 0)
            ci_2020 = bias_2020[uid].get('CI_out', 0)
            rank_2016 = bias_2016[uid].get('rank', np.nan)
            rank_2020 = bias_2020[uid].get('rank', np.nan)
            bias_stats.append({'user_id': uid, 'CI_2016': ci_2016, 'CI_2020': ci_2020,
                               'rank_2016': rank_2016, 'rank_2020': rank_2020, 'bias': bias})


        #bias_user_rankings[2016][bias] = bias_2016
        #bias_user_rankings[2020][bias] = bias_2020

        bias_stats = pd.DataFrame(bias_stats)
        bias_stats.user_id = bias_stats.user_id.astype(int)

        #bias_stats.CI_2016 /= np.max(bias_stats.CI_2016)
        #bias_stats.CI_2020 /= np.max(bias_stats.CI_2020)
        #bias_stats.CI_2016 /= np.sum(bias_stats.CI_2016)
        #bias_stats.CI_2020 /= np.sum(bias_stats.CI_2020)

        #bias_stats.loc[bias_stats.rank_2016 > 50, 'rank_2016'] = 51
        #bias_stats.loc[bias_stats.rank_2020 > 50, 'rank_2020'] = 51
        bias_stats['delta_rank'] = bias_stats.rank_2016 - bias_stats.rank_2020
        bias_stats['delta'] = (bias_stats.CI_2020/bias_stats.CI_2020.sum()) - (bias_stats.CI_2016/bias_stats.CI_2016.sum())
        bias_stats = bias_stats[bias_stats.user_id.isin(set(user_map.keys()))]
        bias_stats['user_handle'] = bias_stats.user_id.apply(apply_user_map)
        bias_stats['verified'] = bias_stats.user_id.apply(verified_map)
        bias_stats = bias_stats.sort_values(by='delta', ascending=False)
        #bias_stats = bias_stats[(bias_stats.delta >= 0.01) | (bias_stats.delta <= -0.01)]

        sankey_data(bias_stats, bias)
        bias_stats.to_csv('results/top_influencers_{}.csv'.format(bias), index=False)
        if plot:
            # rank stuff
            bias_stats = bias_stats[(bias_stats['rank_2020'] >= 30) | (bias_stats['rank_2016'] >= 30)]
            plot_influence_gain(bias_stats, N, 'Largest Change in Influence {}'.format(bias),
                                'delta_influence_stem_{}'.format(bias))


    for uid in set(user_to_ci_2016.keys()) | set(user_to_ci_2020.keys()):
        # we will set the bias of the user to be that with the highest CI
        #bias = max(user_to_ci_2020)
        ci_2016 = user_to_ci_2016.get(uid, 0)
        ci_2020 = user_to_ci_2020.get(uid, 0)
        user_stats.append({'user_id': uid, 'CI_2016': ci_2016, 'CI_2020': ci_2020})

    user_stats = pd.DataFrame(user_stats)
    user_stats.user_id = user_stats.user_id.astype(int)
    #user_stats.CI_2016 /= np.max(user_stats.CI_2016)
    #user_stats.CI_2020 /= np.max(user_stats.CI_2020)
    #user_stats.CI_2016 /= np.sum(user_stats.CI_2016)
    #user_stats.CI_2020 /= np.sum(user_stats.CI_2020)

    user_stats['delta'] = (user_stats.CI_2020 / user_stats.CI_2020.sum()) - (user_stats.CI_2016 / user_stats.CI_2016.sum())

    print("len user_stats before", len(user_stats))
    user_stats = user_stats[user_stats.user_id.isin(set(user_map.keys()))]
    print("len user_stats after", len(user_stats))

    user_stats['user_handle'] = user_stats.user_id.apply(apply_user_map)
    user_stats['verified'] = user_stats.user_id.apply(verified_map)
    user_stats = user_stats.sort_values(by='delta', ascending=False)
    #user_stats = user_stats[(user_stats.delta >= 0.01)|(user_stats.delta <= -0.01)]
    if plot:
        plot_influence_gain(user_stats, N, 'Largest Change in Influence Combined',
                            'delta_influence_stem_{}'.format('combined'))
    user_stats.to_csv('results/top_influencers.csv', index=False)
    sankey_data(user_stats, 'combined')

    # create a csv that has a feature per bias and the rows are users and their normalized delta ci per bias
    # also do a plot where rows are user and CI per bias for each year
    combined_bias_df = pd.DataFrame()
    for year in [2016, 2020]:
        users = set([uid for uid in bias_user_rankings[year].keys()])
        #rows = [{'uid': uid}.update({bias: bias_user_rankings[year].get(uid, 0) for bias in bias_user_rankings[year].keys()})
        # for uid in users]
        rows = [
            {'uid': uid, **{bias: bias_user_rankings[year][uid].get(bias, 0) for bias in BIAS_ENUM.values()}}
            for uid in users
        ]

        user_bias_cis = pd.DataFrame(rows).set_index('uid')
        # remove any users that have no influence in any category
        user_bias_cis = user_bias_cis.loc[(user_bias_cis != 0).any(axis=1)]
        user_bias_cis = user_bias_cis / user_bias_cis.sum()
        user_bias_cis.to_csv('results/igloo_{}.csv'.format(year))
        plt.title("Polarization {}".format(year))
        igloo_plot(user_bias_cis, 'results/igloo_{}.png'.format(year), key_col='uid')
        plt.clf()

    #print("Change in Top {} 2016 user's CI".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[1], reverse=True)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016,  delta))
    #    except KeyError as e:
    #        pdb.set_trace()
#
    #print("Top {} Users by CI gain".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[3], reverse=True)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016, delta))
    #    except KeyError as e:
    #        pdb.set_trace()
#
    #print("Top {} users by CI loss".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[3], reverse=False)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016, delta))
    #    except KeyError as e:
    #        pdb.set_trace()
    return user_stats


def plot_influence_gain(delta_influence_df, N, title='Largest Change in Influence', filename='delta_influence_stem'):
    print("Plotting influence game {}".format(filename))
    #top_bot_ten = delta_influence_df.nlargest(10, 'delta').append(delta_influence_df.nsmallest(10, 'delta'))
    top_ten = delta_influence_df.nlargest(N, 'delta_rank')
    bot_ten = delta_influence_df.nsmallest(N, 'delta_rank').sort_values(by='delta_rank', ascending=False)
    #top_ten = delta_influence_df.nlargest(N, 'delta')
    #bot_ten = delta_influence_df.nsmallest(N, 'delta').sort_values(by='delta', ascending=False)

    # get the top 10 losses and top N gains
    plt.stem(top_ten.user_handle, top_ten.delta_rank, 'g', markerfmt='go', label='Largest CI gains')
    plt.stem(bot_ten.user_handle, bot_ten.delta_rank, 'r', markerfmt='ro', label='Largest CI losses')
    plt.xticks(#[x for x in top_ten.user_handle]+[x for x in bot_ten.user_handle],
               rotation='vertical', fontsize=10)

    plt.title(title)
    plt.ylabel(r'min-max normalized $\Delta$ CI')
    plt.xlabel('User handle')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.ylim((-1, 1))
    plt.savefig('results/{}.png'.format(filename))
    plt.clf()

    return


def sankey_data(df, bias):
    # TRY WITH FLOW DIAGRAM SANKEY
    data = pd.DataFrame()
    # data['user_handle'] = delta_influence_df.user_handle
    # data['rank_2016'] = delta_influence_df.rank_2016
    # data['rank_2020'] = delta_influence_df.rank_2020
    data['source'] = df.user_handle
    data['target'] = df.user_handle
    # data['value'] =

    # plan, plot the flow of influencers to and from percentile influencer categories.
    # see where influencers from the top 30 are dropping to and what percentiles the new influencers are
    # coming from, by bias category.

    # calculate percentiles
    # source: 2016 percentile, target: 2020 percentile, user_handle: for fun, type: bias, value: 1
    percentiles = [25, 50, 75, 80, 85, 90, 95, 99]
    percentiles_CI_2016 = np.percentile(np.array(df[df.CI_2016 != 0].CI_2016), percentiles)
    percentiles_CI_2020 = np.percentile(np.array(df[df.CI_2020 != 0].CI_2020), percentiles)

    percentile_classes_2016 = []

    data['user_handle'] = df['user_handle']
    data['CI_2016'] = df.CI_2016
    data['CI_2020'] = df.CI_2020
    data['source'] = np.ones_like(df.user_handle)
    data['target'] = np.ones_like(df.user_handle)

    data.loc[data.CI_2016 == 0, 'source'] = 'no influence'
    data.loc[data.CI_2016 <= percentiles_CI_2016[0], 'source'] = 'lowest 25%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[0], 'source'] = 'top 75%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[1], 'source'] = 'top 50%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[2], 'source'] = 'top 25%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[3], 'source'] = 'top 20%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[4], 'source'] = 'top 15%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[5], 'source'] = 'top 10%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[6], 'source'] = 'top 5%'
    data.loc[data.CI_2016 >= percentiles_CI_2016[7], 'source'] = 'top 1%'

    data.loc[data.CI_2020 == 0, 'target'] = 'no influence'
    data.loc[data.CI_2020 <= percentiles_CI_2020[0], 'target'] = 'lowest 25%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[0], 'target'] = 'top 75%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[1], 'target'] = 'top 50%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[2], 'target'] = 'top 25%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[3], 'target'] = 'top 20%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[4], 'target'] = 'top 15%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[5], 'target'] = 'top 10%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[6], 'target'] = 'top 5%'
    data.loc[data.CI_2020 >= percentiles_CI_2020[7], 'target'] = 'top 1%'

    data['value'] = np.ones_like(df.user_handle)

    data.to_csv('results/sankey_data_{}.csv'.format(bias))


    return


def sankey_plot():
    import pandas as pd
    import ipysankeywidget
    import floweaver as fw

    flows = pd.read_csv('sankey_data_right.csv')
    add_text = lambda df, text: df + text
    flows.source = flows.source.apply(add_text, args=(' 2016',))
    flows.target = flows.target.apply(add_text, args=(' 2020',))

    flows = flows.groupby(['source', 'target'])[['source', 'target', 'value']].sum().reset_index()

    size = dict(width=570, height=300)
    nodes = {
       '2016': fw.ProcessGroup(['lowest 25%', 'top 75%', 'top 50%', 'top 25%',
                                'top 20%', 'top 15%', 'top 10%', 'top 5%', 'top 1%']),
       '2020': fw.ProcessGroup(['lowest 25%', 'top 75%', 'top 50%', 'top 25%',
                                'top 20%', 'top 15%', 'top 10%', 'top 5%', 'top 1%'])
    }

    ordering = [
       ['2016'],
       ['2020']
    ]

    bundles = [fw.Bundle('2016', '2020')]
    sdd = fw.SankeyDefinition(nodes, bundles, ordering)
    fw.weave(sdd, flows).to_widget(**size)

    return


def igloo_plot(df, out_path, key_col='user_handle'):
    """
    Implements the IglooPlot from this paper:
        Kuntal, Bhusan K., Tarini Shankar Ghosh, and Sharmila S. Mande. "Igloo-Plot: a tool for visualization of
        multidimensional datasets." Genomics 103.1 (2014): 11-20.

    :param df:
    :param outpath:
    :return:
    """
    # step 1) load (taken care of)
    # step 2) normalization

    # step 3) pair-wise correlation
    # to create our matrix, we need to loop over each pair of features and do distance correlation
    correlation_matrix = []
    row_col_names = []
    for f1 in df.columns:
        if f1 == key_col:
            continue

        row_col_names.append(f1)
        correlation_row = []
        for f2 in df.columns:
            if f2 == key_col:
                continue

            correlation_row.append(correlation(df[f1], df[f2], centered=True))

        correlation_matrix.append(correlation_row)
    #df = df.set_index(key_col)
    pd_correlation_matrix = df.corr(method='pearson')

    # step 4) feature arrangement
    # we will try two arrangement styles, the first will be static arrangement, where we arrange the biases as
    # we do elsewhere in the paper, far left, left, lean left, center, lean right, right, far right, fake
    # the other arrangement will be how the paper does it, which is to correlate the features and arrange them
    # according to distance between features. This likely encodes more nice information on the dataset
    #first_sample = pd_correlation_matrix.sample()
    #heirarchy = {first_sample.index: first_sample}

    dissimilarity = 1 - pd_correlation_matrix
    hierarchy = linkage(squareform(dissimilarity), method='average')
    threshold = 0.01
    step = 0.001
    heir_clusters = []
    unique_partitions = []
    while True:
        labels = fcluster(hierarchy, threshold, criterion='distance')
        if tuple(labels) not in unique_partitions:
            unique_partitions.append(tuple(labels))
            heir_clusters.append([set(np.where(labels == i)[0]) for i in np.unique(labels)])

        if len(labels) == 1 or threshold >= 1:
            break

        threshold += step

    ordered = place_nodes(heir_clusters[-1][0], -1, heir_clusters)
    ordered_biases = pd_correlation_matrix.columns[ordered]
    # manual ordered biases
    #ordered_biases = ['left extreme', 'left', 'left leaning', 'center', 'right leaning', 'right extreme', 'right', 'fake']
    ordered_biases = ['left extreme', 'left', 'left leaning', 'center', 'right leaning', 'right', 'right extreme',
                      'fake']
    positions = [0]
    for b1, b2 in zip(ordered_biases[:-1], ordered_biases[1:]):
        positions.append(positions[-1] + dissimilarity[b1][b2])
    #positions = [0] + [dissimilarity[b1][b2] for b1, b2 in zip(ordered_biases[:-1], ordered_biases[1:])]

    #far_left_idx = np.where(dissimilarity.columns == 'left extreme')[0][0]
    #for partition in heir_clusters[::-1][:-1]:

    # step 5) semi-circular transform
    radius = np.max(positions)/2
    midpoint = (radius, 0)

    bias_positions = {}
    angles = [radians(x) for x in np.linspace(-90, 90, 8)]
    #angles = [x if x < 90 else abs(x-180) for x in angles]
    #for bias, x in zip(ordered_biases, positions):
    for angle, bias in zip(angles, ordered_biases):
        #distance = radius - x
        #angle = (pi * distance) / (2 * radius)
        #angle = (i * (180 / len(ordered_biases)))

        un = np.abs(radius * np.sin(angle))
        vn = radius*(1-np.cos(angle))
        #vn = radius*np.cos(angle)

        print("bias: {}, radius: {}, un: {}, radius - un: {}".format(bias, radius, un, radius-un))
        #if x > radius:
        if angle > 0:
            #bias_positions[bias] = np.array([radius + vn, radius - un])
            bias_positions[bias] = np.array([radius + un, radius - vn])
        else:
            #bias_positions[bias] = np.array([radius - vn, radius - un])
            bias_positions[bias] = np.array([radius - un, radius - vn])
       # bias_positions[bias] = np.array([x, np.sqrt(radius**2 - distance**2)])

    xs = np.linspace(0, np.max(positions), num=1000)
    ys = np.linspace(0, radius, num=1000)
    X, Y = np.meshgrid(xs, ys)
    F = X**2 + Y**2
    semicircle = np.array([np.sqrt(radius**2 - (radius - x)**2) for x in xs])

    # step 6) datapoint projection
    transformed_datapoints = np.array([
        np.sum([df.loc[x][bias] * bias_vector for bias, bias_vector in bias_positions.items()], axis=0) / np.sum(df.loc[x])
        for x in df.index
    ])

    # finally we plot all the things
    plt.plot(xs, semicircle)
    #plt.contour(X, Y, F, [0])
    bias_position_x = [x[0] for x in bias_positions.values()]
    bias_position_y = [x[1] for x in bias_positions.values()]
    plt.scatter(bias_position_x, bias_position_y, s=100, facecolors='w', edgecolors='r')
    plt.scatter(transformed_datapoints.T[0], transformed_datapoints.T[1], c='g')
    for bias, position in bias_positions.items():
        plt.annotate(bias, (position[0], position[1]))

    #plt.title('Influencer Polarization')
    plt.savefig(out_path)

    return


def place_nodes(partition, pos, heirarchy):
    if pos == 0 or pos == -(len(heirarchy)):
        return partition

    out_nodes = []
    if partition in heirarchy[pos]:
        out_nodes.extend(place_nodes(partition, pos-1, heirarchy))
    else:
        for cluster in heirarchy[pos]:
            if cluster.issubset(partition):# in partition:
                out_nodes.extend(place_nodes(cluster, pos-1, heirarchy))

    return out_nodes


def plot_bias_nodes_edges(stats_2016, stats_2020):
    rows = [dict({'bias': key}, **val)for key, val in stats_2016.items()]
    df_2016 = pd.DataFrame(rows)
    rows = [dict({'bias': key}, **val) for key, val in stats_2020.items()]
    df_2020 = pd.DataFrame(rows)

    df_2016['fractional_nodes'] = np.array(df_2016.n_nodes) / np.sum(df_2016.n_nodes)
    df_2016['fractional_edges'] = np.array(df_2016.n_edges) / np.sum(df_2016.n_edges)

    df_2020['fractional_nodes'] = np.array(df_2020.n_nodes) / np.sum(df_2020.n_nodes)
    df_2020['fractional_edges'] = np.array(df_2020.n_edges) / np.sum(df_2020.n_edges)

    compare_df = df_2016[['bias', 'fractional_nodes']].rename(columns={'fractional_nodes': '2016'}).merge(
        df_2020[['bias', 'fractional_nodes']].rename(columns={'fractional_nodes': '2020'}),
        on=['bias']).set_index('bias')

    #ordered_biases = ['fake', 'right extreme', 'right', 'right leaning',
    #                  'center', 'left leaning', 'left', 'left extreme']
    ordered_biases = ['left extreme', 'left', 'left leaning', 'center',
                      'right leaning', 'right', 'right extreme', 'fake']
    #fig, ax = plt.subplots()
    compare_df.loc[ordered_biases].plot.bar(rot=45)
    #df_2020.plot.bar('bias', 'n_nodes', ax=ax)
    plt.ylabel('Node count')
    plt.xlabel('Bias Network')
    plt.title('Bias network node ratios 2016 vs 2020')
    plt.tight_layout()
    plt.savefig('results/bias_network_node_compare.png')
    plt.clf()

    # edges plot
    compare_df = df_2016[['bias', 'fractional_edges']].rename(columns={'fractional_edges': '2016'}).merge(
        df_2020[['bias', 'fractional_edges']].rename(columns={'fractional_edges': '2020'}),
                                                             on=['bias']).set_index('bias')
    #fig, ax = plt.subplots()
    compare_df.loc[ordered_biases].plot.bar(rot=45)
    #df_2020.plot.bar('bias', 'n_edges', ax=ax)
    plt.ylabel('Edge count')
    plt.xlabel('Bias Network')
    plt.title('Bias network edge ratios 2016 vs 2020')
    plt.tight_layout()
    plt.savefig('results/bias_network_edge_compare.png')
    plt.clf()
    return


def all_network_stats():
    graphs_path = '/home/crossb/packaged_ci/graphs/2020/'
    write_path = '/home/crossb/research/elites_2020/information_diffusion-master/results'
    stats_by_year = {}
    completeness = 'simple' #'complete'
    for path, year in [('/home/crossb/packaged_ci/graphs/2016/', '2016'),
                       ('/home/crossb/packaged_ci/graphs/2020/', '2020')
                       ]:
        graphs_gt = load_graphs_gt(path, year, completeness=completeness)
        network_stats = {}
        for bias, graph in graphs_gt.items():
            samples = 78911 if year == '2016' else 78911#21411
            network_stats[bias] = network_characteristics_gt(graph, samples)

        stats_by_year[year] = network_stats

        # write to csv
        with open(os.path.join(write_path, 'network_characteristics_{}_{}.csv'.format(year, completeness)), 'w') as outfile:
            write_header = True
            for bias, characteristics in network_stats.items():
                if write_header:
                    print('{},{},{},{},{},{},{},{},{}'.format('', *characteristics.keys()), file=outfile)
                    write_header = False
                print(','.join([bias] + [str(value) for stat, value in characteristics.items()]), file=outfile)

    return


def read_edgeslist(path, columns=('auth_id', 'infl_id')):
    edges = dd.read_csv(path, delimiter=',',
                        usecols=columns,
                        dtype={col: np.int64 for col in columns}
                        )
    return edges


def main():
    print("Default number of threads:", nk.getCurrentNumberOfThreads())
    nk.setNumberOfThreads(64)
    print("Updated number of threads:", nk.getCurrentNumberOfThreads())
    graphs_2020 = load_graphs_gt('/home/crossb/packaged_ci/graphs/2020/', year='2020')
    graphs_2016 = load_graphs_gt('/home/crossb/packaged_ci/graphs/2016/', year='2016')

    delta_df = influence_gain(graphs_2016, graphs_2020, N=15, plot=True)
    #delta_df = influence_gain(graphs_2016, graphs_2020, N=100, plot=False)
    #influencer_network_anlaysis(year=2016)
    #influencer_network_anlaysis(year=2020)
    #all_network_stats()

    return


if __name__ == '__main__':
    main()