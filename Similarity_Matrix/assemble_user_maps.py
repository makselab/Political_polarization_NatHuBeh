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


print('Assembling 2020 user map')

import pickle
from os import listdir
from os.path import isfile, join
import sys

user_map = {}

target_dir = '/path/to/raw/users/'

fnames = [f for f in listdir(target_dir) if isfile(join(target_dir, f))]

for fname in fnames:
    if '.csv' in fname:
        print('Reading', fname)

        with open(target_dir + fname, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n')
                line = line.split(',')

                # print(line)

                try:
                    id = int(line[0])
                    verified = int(line[-1])

                    created_at = str(line[1])
                    name = str(line[-2])
                        
                    user_map[id] = {'created_at': created_at, 'name': name, 'verified': verified}
                except:
                    continue

print('Map size:', len(user_map))
print(25073877, 'is', user_map[25073877])

pickle.dump(user_map, open('maps/user_map_2020.pkl', 'wb'))

print('Assembling 2016 user map')

import sqlite3

user_map = {}

conn = sqlite3.connect('/path/to/2016_election_sqlite3/data/complete_trump_vs_hillary_db.sqlite')
c = conn.cursor()

for row in c.execute('SELECT * FROM influencer_rank_date'):
    # print(row)

    id = int(row[5])
    created_at = str(row[4])
    name = str(row[6]).replace('@', '')

    if '???' not in name:
        user_map[id] = {'created_at': created_at, 'name': name}

print('Map size:', len(user_map))
print(25073877, 'is', user_map[25073877])

pickle.dump(user_map, open('maps/user_map_2016.pkl', 'wb'))