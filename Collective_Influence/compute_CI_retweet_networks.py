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


import sys
import os

import CIcython # Use Alexandre Bovet's edited CI code

import time
import pickle
import graph_tool.all as gt

import numpy as np

def format_graph_file(file, year):
    file = file.split('/')
    file = file[-1]
    file = file.split('.')
    return '/path/to/generate_ci/graphs/' + str(year) + '/' + file[0] + '_ci.gt'

def format_pickle_file(file, year, direction):
    file = file.split('/')
    file = file[-1]
    file = file.split('.')
    return '/path/to/generate_ci/output/' + str(year) + '/' + file[0] + '_' + direction + '_ci.pkl'

def add_CI_to_graph(file, year):
    
    print('pid ' + str(os.getpid()) + ' loading file ' + file)
    
    graph = gt.load_graph(file)
        
    for direction in ['out', 'in']:
        
        print('pid ' + str(os.getpid()) + ' -- ' + direction)
        t0 = time.time()
        CIranks, CImap = CIcython.compute_graph_CI(graph, rad=2,
                                          direction=direction,
                                          verbose=True)
        t1 = time.time() - t0
        
        print('pid ' + str(os.getpid()) + ' -- ' + str(t1))
        
        print('pid ' + str(os.getpid()) + ' saving CIranks ' + direction +  '_' + file.strip('.gt'))
        with open(format_pickle_file(file, year, direction), 'wb') as fopen:
            pickle.dump({'CIranks': CIranks, 'CImap' : CImap, 'time' : t1}, fopen)
    
        graph.vp['CI_' + direction] = graph.new_vertex_property('int64_t', vals=0)
        graph.vp['CI_' + direction].a = CImap
        
    print('pid ' + str(os.getpid()) + ' -- adding katz centrality')
    
    graph.vp['katz'] = gt.katz(graph)
    
    graph.set_reversed(True)
    graph.vp['katz_rev'] = gt.katz(graph)
    graph.set_reversed(False)
        
    graph_file = format_graph_file(file, year)

    print('pid ' + str(os.getpid()) + ' saving graph ' + graph_file)
    graph.save(graph_file)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        fname = sys.argv[1]
        year = sys.argv[2]

        add_CI_to_graph(fname, year)