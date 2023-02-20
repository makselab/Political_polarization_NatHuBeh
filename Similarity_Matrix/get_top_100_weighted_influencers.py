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
import networkit as nk
import numpy as np
import pickle

from os import listdir
from os.path import isfile, join

import pandas as pd

top_n = 100

years = [2016, 2020]

centrality_measure = 'PageRank' # PageRank or Katz

for year in years:
    target_dir = '/path/to/edge_lists/url_classified_edgelists/' + str(year) + '/'
    map_dir = '/path/to/maps/url_classified_edgelists/' + str(year) + '/'

    fnames = [f for f in listdir(target_dir) if isfile(join(target_dir, f))]

    data = {'rank': list(range(1, top_n + 1))}
    for fname in fnames:
        if ".txt" in fname:
            print('Loading', fname)

            oname = '/path/to/weighted//top_100_' + fname.replace('.txt', '') + '_' + centrality_measure.lower() + '.pkl'

            print('Output:', oname)

            map_res = pickle.load(open(map_dir + fname.replace('.txt', '.pkl'), 'rb'))
            node_map = map_res['node_map']
            reverse_node_map = map_res['reverse_node_map']
            edgeListReader = nk.graphio.EdgeListReader(',', 0, '#', directed=True)
            G = edgeListReader.read(target_dir + fname)

            if centrality_measure == 'PageRank':
                cm = nk.centrality.PageRank(G, 1e-6)
            elif centrality_measure == 'Katz':
                cm = nk.centrality.KatzCentrality(G, 1e-3)
            cm.run()
            rankings = cm.ranking()

            res = {}
            rank = -1
            for id, score in rankings: 
                if rank + 1 >= top_n:
                    break
                else:
                    rank += 1

                user_id = reverse_node_map[id]
                res[user_id] = rank + 1

            pickle.dump(res, open(oname, 'wb'))