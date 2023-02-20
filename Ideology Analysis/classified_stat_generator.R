# Copyright (C) 2021-2026, Alessandro Galeazzi

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

##### LOAD ALL CLASSIFIED LINKS WE HAVE #####
library(data.table)
library(dplyr)
#load("clasified_tweets.Rdata")
load("classified_data_with_doubles.Rdata")
files=list.files("unofficial_client_urls", full.names = T)

unofficial_tweet_ids=list()
for (file in files)
{
  print(file)
  un_data=fread(file,header=T)
  unofficial_tweet_ids=c(unofficial_tweet_ids,list(un_data))
}

unofficial_tweet_ids=rbindlist(unofficial_tweet_ids)
unofficial_tweet_ids=distinct(unofficial_tweet_ids,tweet_id,.keep_all=T)
unofficial_tweet_ids$tweet_id=as.character(unofficial_tweet_ids$tweet_id)
unofficial_tweet_ids$type="unofficial"
classified=classified_data_with_double_links
classified$tweet_id=as.character(classified$tweet_id)
classified=merge(classified,unofficial_tweet_ids[,.(tweet_id,type)],
                 by.x="tweet_id",
                 by.y="tweet_id",
                 all.x=T)

classified_stats=classified[,.(n_tweet=.N, 
                               p_tweet=.N/nrow(classified),
                               n_user=length(unique(user_id)),
                               p_user=length(unique(user_id))),
                               by=.(bias)]
classified_stats$p_user=classified_stats$p_user/sum(classified_stats$p_user)

classified_stats_by_type=classified[,.(n_tweet=.N, 
                                       n_user=length(unique(user_id))),
                                      by=.(bias, type)]

save(classified_stats, file="classified_stats_with_double_links.Rdata")
save(classified_stats_by_type, file="classified_stats_by_type_double.Rdata")
domains_sanity_check=classified[,.(count=.N),by=.(bias,domain)]
save(domains_sanity_check, file="domains_sanity_check.Rdata")

