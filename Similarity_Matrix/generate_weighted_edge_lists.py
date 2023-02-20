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

import numpy as np
import pickle
import os.path
from os import path
import collections
import pandas as pd

def csv_generate_edge_list(year, stance, weighted=False):
    print('Using CSV loader')
    fname = '/path/to/url_classified_edgelists/' + str(year) + '/' + stance + '_' + str(year) + '.csv'

    print('Generating edge list for', fname)
    if not path.exists('/path/to/maps/url_classified_edgelists/' + str(year) + '/' + stance + '_' + str(year) + '.pkl'):
        print('Generating node maps')
        nodes = {}

        if year == 2020:
            first_line = True # Skip header
        elif year == 2016:
            first_line = False

        with open(fname, 'r') as ins:
            for row in ins:
                if first_line:
                    first_line = False
                    continue

                row = row.split(',')
                if year == 2020:
                    id = int(row[0]) # tid
                    source = int(row[1]) # auth_id
                    target = int(row[2]) # infl_id
                elif year == 2016:
                    id = int(row[2]) # tid
                    source = int(row[1]) # auth_id
                    target = int(row[0]) # infl_id

                try:
                    nodes[source] += 1
                except:
                    nodes[source] = 1

                try:
                    nodes[target] += 1
                except:
                    nodes[target] = 1

        nodes = list(nodes.keys())

        print('Unique node count:', len(nodes))

        node_map = {}
        reverse_node_map = {}
        count = 0
        for node in nodes:
            node_map[node] = count
            reverse_node_map[count] = node
            count += 1

        map_res = {'node_map': node_map, 'reverse_node_map': reverse_node_map}
        pickle.dump(map_res, open('/path/to/maps/url_classified_edgelists/' + str(year) + '/' + stance + '_' + str(year) + '.pkl', 'wb'))
    else:
        print('Loading node maps')
        map_res = pickle.load(open('/path/to/maps/url_classified_edgelists/' + str(year) + '/' + stance + '_' + str(year) + '.pkl', 'rb'))
        node_map = map_res['node_map']
        reverse_node_map = map_res['reverse_node_map']

    print('Gathering edge list')
    
    if year == 2020:
        first_line = True # Skip header
    elif year == 2016:
        first_line = False

    edge_list = {}
    with open(fname, 'r') as ins:
        for row in ins:
            if first_line:
                first_line = False
                continue

            row = row.split(',')
            
            if year == 2020:
                source = node_map[int(row[1])] # auth_id
                target = node_map[int(row[2])] # infl_id
            elif year == 2016:
                source = node_map[int(row[1])] # auth_id
                target = node_map[int(row[0])] # infl_id
            
            try:
                edge_list[(source, target)] += 1
            except:
                edge_list[(source, target)] = 1

    print('Number of', stance, 'stance edges:', len(edge_list))

    print('Sorting edge list')
    edge_list = dict(collections.OrderedDict(sorted(edge_list.items())))

    print('Saving edge list')
    w = open('/path/to/edge_lists/url_classified_edgelists/' + str(year) + '/' + stance + '_' + str(year) + '.txt', 'w')
    for edge in edge_list:
        weight = edge_list[edge]
        source, target = edge
        if weighted:
            w.write(str(source) + ',' + str(target) + ',' + str(weight) + '\n')
        else:
            w.write(str(source) + ' ' + str(target) + '\n')
    w.close()

if __name__ == '__main__':
    csv_generate_edge_list(2016, 'center', weighted = True)
    csv_generate_edge_list(2016, 'extreme_bias_left', weighted = True)
    csv_generate_edge_list(2016, 'extreme_bias_right', weighted = True)
    csv_generate_edge_list(2016, 'fake', weighted = True)
    csv_generate_edge_list(2016, 'lean_left', weighted = True)
    csv_generate_edge_list(2016, 'lean_right', weighted = True)
    csv_generate_edge_list(2016, 'left', weighted = True)
    csv_generate_edge_list(2016, 'right', weighted = True)