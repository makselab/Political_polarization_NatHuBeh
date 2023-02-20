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
unofficial_tweet_ids=fread("unofficial_tweet_ids.csv", header=F,colClasses = "character")
unofficial_tweet_ids=rename(unofficial_tweet_ids,tweet_id=V1)
unofficial_tweet_ids=distinct(unofficial_tweet_ids,tweet_id,.keep_all=T)
unofficial_tweet_ids$tweet_id=as.character(unofficial_tweet_ids$tweet_id)
unofficial_tweet_ids$type="unofficial"
classified=classified_data_with_double_links
classified$tweet_id=as.character(classified$tweet_id)
classified=merge(classified,unofficial_tweet_ids[,.(tweet_id,type)],
                 by.x="tweet_id",
                 by.y="tweet_id",
                 all.x=T)
classified$type[is.na(classified$type)]="official"

usr_stat=classified[,.(count=.N,
                     type=ifelse("unofficial" %in% type, "unofficial","official")),
                  by=.(user_id, bias)]

usr_stat_unique=usr_stat[,.SD[which.max(rank(count, ties.method = "random"))],by=.(user_id)]
usr_stat_by_cat=merge(usr_stat_unique[,.(N_u=.N), by=bias],usr_stat_unique[type=="unofficial",.(Nu_no=.N), by=bias],by = "bias")
table_data=merge(classified[,.(N_t=.N), by=bias],classified[type=="unofficial", .(Nt_no=.N), by=bias], by.x="bias", by.y="bias")
table_data=merge(table_data,usr_stat_by_cat,by="bias")

table_data$bias=factor(table_data$bias, 
                       levels = c("Fake news","Extreme bias right","Right news","Right leaning news",
                                  "Center news",
                                  "Left leaning news","Left news","Extreme bias left"))
table_data=table_data[order(bias)]
table_data$p_t=table_data$N_t/sum(table_data$N_t)
table_data$p_u=table_data$N_u/sum(table_data$N_u)
table_data$Nt_Nu=table_data$N_t/table_data$N_u
table_data$pt_no=table_data$Nt_no/table_data$N_t
table_data$pu_no=table_data$Nu_no/table_data$N_u
table_data$Ntno_Nuno=table_data$Nt_no/table_data$Nu_no

table_data_for_plot_2020=table_data[,.(bias,N_t,p_t,N_u,p_u,Nt_Nu,pt_no,pu_no,Ntno_Nuno)]
save(table_data_for_plot_2020, file="table_data_unique_users_2020.Rdata")
