Here you can find the Data and codes needed to reproduce the results from: "Political polarization of news media and influencers on Twitter in the 2016 and 2020 US presidential elections"

Refer to the following link for the data:

https://osf.io/e395q

Refer to the following link for the codes: 
https://osf.io/dbzm2/

################
ABOUT THE DATA
################

This dataset contains the classified tweets, the tweet ids, and the retweet netwoks used in the article:

"Political polarization of news media and influencers on Twitter in the 2016 and 2020 US presidential elections". Authors: James Flamino, Alessandro Galezzi, Stuart Feldman, Michael W. Macy, Brendan Cross, Zhenkun Zhou, Matteo Serafino, Alexandre Bovet, Hernan A. Makse , Boleslaw K. Szymanski.

The classification of news outlets in the different media categories is a matter of opinion, rather than a statement of fact. This opinion originated in publicly available datasets from fact-checking organizations, i.e. www.opensources.co (copy at https://github.com/alexbovet/opensources), www.mediabiasfactcheck.com & www.allsides.com. This classification of news media should not be interpreted as representing the opinions of the authors of the article.

There are 4 folders.

Classified_Tweets: Contains the lists of tweet IDs of the tweets we were able to classify through the link they contain and the corresponding news outlet category.

Influencers_Classification: Contains the classication of the most important influcencers into a category.

Retweet_Networks: The retweets_network_final.tar.gz folder contains 8 retweet network csv files, one for each news category. All the entries in those csv's correspond to a retweet of a tweet classified as the csv's news category. These files are edgelists, with each retweet being a directed edge from nodes infl-id to auth-id. The columns are as follows: (id), i.e. the id of the retweet, (auth-id), i.e. the user id of the user who authored the retweet (the user who is being influenced), and (infl-id), i.e. the user id of the influencer, the one who wrote the original tweet that is now being retweeted.

center_retweet_edges.csv
fake_retweet_edges.csv
left_extreme_retweet_edges.csv
left_leaning_retweet_edges.csv
left_retweet_edges.csv
right_extreme_retweet_edges.csv
right_leaning_retweet_edges.csv
right_retweet_edges.csv
Tweet_IDs: The tweet_ids_2016.txt.zst file contains the ID of tweets, retweets, and quotes for the 2016 election.

Google Drive: PolarizationUSElections: The tweet_ids_2020.txt.zst file contains the ID of tweets, retweets, and quotes for the 2020 election.

The media categories and the news outlets in each category are detailed in our article.

The retweet networks contain parallel edges whenever a user retweeted another user more than once.

Softwares such as hydrator and tweepy can be used to “rehydrate” the tweet_IDs, i.e. download the full tweet objects using the tweet_IDs.
