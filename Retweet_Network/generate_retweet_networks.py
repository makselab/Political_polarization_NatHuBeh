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

####################################################################################################
#
# This script is responsible for generating the retweet networks using the raw tweet / retweet data.
#
####################################################################################################

from dask.distributed import Client, LocalCluster, progress
import dask.dataframe as dd
import time
import pickle
import graph_tool.all as gt
from multiprocessing import Pool
from datetime import datetime
import os
import pandas as pd
import numpy as np


# MODIFY BASED ON YOUR SYSTEM
NUM_WORKERS = 32
THREADS_PER_WORKER = 1
MAX_MEMORY_USED = '256GB'
WORKING_DIR = '~/working'
WRITE_DIR = 'out/retweet_networks'

# Start and end date of our considered tweet time range
start_date = datetime(2020, 6, 1)
stop_date = datetime(2020, 11, 3)

# home of the raw tweets, retweets, quotes, mentions, etc. Modify accordingly
Election_2020_dir = '/home/pub/hernan/Election_2020/joined_output_v2'
CLASSIFICATIONS_PATH = '/home/pub/hernan/Election_2020/classified_tweets_v3.csv'


def dask_str_col_to_int(data, col):
    to_numeric = lambda df: pd.to_numeric(df, errors='coerce')
    data = data[~data[col].map_partitions(to_numeric).isna()]
    data[col] = data[col].astype(int)
    return data


def gather_retweet_edges_v2(tweet_dir, retweet_dir):
    """
    This method loads tweets and retweets, merges and then saves edgelists as intermediary files.
    The tweets data contains the timestamp that we desire to filter retweet data on.
    :return:
    """

    # load tweets
    tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',', usecols=['tweet_id', 'created_at'],
                             dtype={'tweet_id': str, 'created_at': str}
                             ).rename(columns={'tweet_id': 'id',  'created_at': 'timestamp'})

    # convert id to int and set as index
    tweet_data = tweet_data[tweet_data['id'] != 'id']
    tweet_data = dask_str_col_to_int(tweet_data, 'id').persist(retries=100)
    progress(tweet_data)
    print("Tweets: Convert id to int")

    tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    filter_dates = lambda df: df[(df['timestamp'] > start_date) & (df['timestamp'] < stop_date)]
    tweet_data = tweet_data.map_partitions(filter_dates)
    tweet_data = tweet_data.persist()
    progress(tweet_data); print("Tweets: Filtered tweets on datetime")

    # remove duplicates
    partitions = tweet_data.npartitions
    tweet_data = tweet_data.drop_duplicates(subset='id').repartition(npartitions=partitions).persist(retries=100)
    progress(tweet_data); print("Tweets: Dropped duplicates")

    # set index
    tweet_data = tweet_data.set_index('id').persist(retries=100)
    progress(tweet_data); print("Tweets: Set index to id")


    # load retweets
    retweet_data = dd.read_csv(os.path.join(retweet_dir, '*.csv'), delimiter=',',
                               usecols=['tweet_id', 'retweet_id', 'auth_id', 'infl_id'],
                               dtype={'tweet_id': str, 'retweet_id': str, 'auth_id': str, 'infl_id': str}
                               ).rename(columns={'tweet_id': 'id', 'user_id': 'auth_id'})

    # convert string columns to int
    retweet_data = dask_str_col_to_int(retweet_data, 'id').persist(retries=100)
    progress(retweet_data); print("Retweets: Converted id to int")

    # remove duplicates
    partitions = retweet_data.npartitions
    retweet_data = retweet_data.drop_duplicates(subset='id').repartition(npartitions=partitions).persist(retries=100)
    progress(retweet_data); print("Retweets: Dropped duplicates")

    # set index
    retweet_data = retweet_data.set_index('id').persist(retries=100)
    progress(retweet_data); print("Retweets: Set index to id")

    # join the data
    print("Merging tweets to retweets")
    intermediaries = retweet_data.merge(tweet_data, left_index=True, right_index=True).persist(retries=100)
    progress(intermediaries); print("Merged tweets and retweets")

    print("Writing Filtered Retweet intermediaries")
    write_dir = os.path.join(WORKING_DIR, 'stance_merged_retweets_updated')
    intermediaries.to_csv(os.path.join(write_dir, 'merged_retweets_alldata_*.csv'))

    print("Done with the thing")
    return


def load_retweet_intermediaries():
    """
    This method loads retweet edge intermediary data.

    :return:
    """
    path = os.path.join(os.path.join(WORKING_DIR, 'stance_merged_retweets_updated'))
    print("Loading intermediaries from {}".format(path))
    data = dd.read_csv(os.path.join(path, '*.csv'), delimiter=',',
                       usecols=['id', 'retweet_id', 'timestamp', 'auth_id', 'infl_id'],
                       dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64, 'retweet_id': np.int64,
                               'infl_id': np.int64}).set_index('retweet_id').persist(retries=100)
    progress(data); print("Finished loading retweet intermediaries")
    return data


def assign_edge_classes(tweets, name='', write=True):
    """
    Assign bias labels to each of the tweet edges using the tweet classifications.
    :param tweets:
        Retweet edge intermediaries.
    :param name:
        The type of network we are creating, 'retweet', 'quote', etc.
    :param write:
        Write the result to our out dir.
    :return:
    """
    print("Begin assigning edge classes")
    bias_to_filename = {'Center news': 'center', 'Fake news': 'fake', 'Extreme bias right': 'right_extreme',
                        'Extreme bias left': 'left_extreme', 'Left leaning news': 'left_leaning',
                        'Right leaning news': 'right_leaning', 'Left news': 'left', 'Right news': 'right'}
    stime_total = time.time()

    # read the tweet classes
    tweet_classes = dd.read_csv(CLASSIFICATIONS_PATH, delimiter=',',
                            usecols=['tweet_id', 'bias'],
                            dtype={'tweet_id': np.int64, 'bias': str}
                            ).rename(columns={'tweet_id': 'retweet_id', 'bias': 'leaning'}).set_index('retweet_id')

    classes = tweet_classes['leaning'].unique()
    for edge_class in classes:
        print("Begin {} edgelist".format(edge_class))
        stime = time.time()
        class_data = tweet_classes[tweet_classes['leaning'] == edge_class].persist(retries=1000)
        #class_data = class_data
        progress(class_data)
        print("Split urls by leaning")
        # merge
        class_data = tweets.merge(class_data, left_index=True, right_index=True).persist(retries=1000)
        progress(class_data)

        if write:
            print("Merged Tweets to urls")
            drop_cols = ['timestamp', 'leaning', 'retweet_id']#, 'p']
            class_data = class_data.reset_index().drop(drop_cols, axis=1)

            print("Writing edgelist to {}".format(WRITE_DIR))
            if not os.path.isdir(WRITE_DIR):
                os.makedirs(WRITE_DIR, exist_ok=True)
            class_data.to_csv(os.path.join(WRITE_DIR, '{}_{}_edges.csv'.format(bias_to_filename[edge_class], name)),
                              single_file=True, index=False)
            print("{} edgelist elapsed time {} seconds".format(edge_class, time.time() - stime))

    print("Assign edge classes elapsed time {} seconds".format(time.time() - stime_total))
    return class_data


def gather_edges():
    print("Begin gathering edge data")
    # load tweets and retweets, filter on date range and merge to get all retweets in date range, save intermediaries.
    gather_retweet_edges_v2(os.path.join(Election_2020_dir, 'tweets'), os.path.join(Election_2020_dir, 'retweets'))

    # load these intermediaries
    edges = load_retweet_intermediaries()

    # Using intermediaries, load tweet classifications and assign all edges a class, then write edgelists to file.
    assign_edge_classes(edges, name='retweet')
    return


def main():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=THREADS_PER_WORKER,
                           scheduler_port=0, dashboard_address=None, memory_limit=MAX_MEMORY_USED)
    client = Client(cluster)
    print("Starting Retweet Network Generation")
    outer_stime = time.time()
    gather_edges()
    print("Generate Retweet Networks Elapsed Time: {} seconds".format(time.time() - outer_stime))
    return


if __name__ == '__main__':
    main()
