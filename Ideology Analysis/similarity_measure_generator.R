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

library(ggplot2)
library(dplyr)
library(data.table)
library(viridis)
library(reshape2)
library(fastmatch)
###### 2020 DATA TO PLOT #####
#### ON THE SERVER #####
#jaccard_coeff=function(x,y)
#{
#  return(sum(x %fin% y)/length(unique(c(x,y))))
#}
load("classified_data_with_doubles.Rdata")
##### COUNT AND EXCLUDE USERS THAT TWEETED JUST ONCE ####
user_stat=classified_data_with_double_links[,.(count=.N),by=user_id]
user_to_keep=user_stat[count>1,user_id]
user_to_keep=classified_data_with_double_links[user_id %fin% user_to_keep]
user_stat=classified_data_with_double_links[,.(count=.N),by=.(user_id, bias)]
usr_stat_unique=user_stat[,.SD[which.max(rank(count, ties.method = "random"))],by=.(user_id)]

biases=unique(usr_stat_unique$bias)

similarity_data=data.table()

for (row in biases)
{
  user=usr_stat_unique[bias==row,user_id]
  for (col in biases)
  {
    count=user_to_keep[user_id %fin% user & bias==col, .N]/user_to_keep[user_id %fin% user, .N]
    n_count=user_to_keep[user_id %fin% user & bias==col, .N]
    similarity_data=rbind(similarity_data,data.table(row=row,col=col,val=count, count=n_count))
  }
}
save(similarity_data, file="similarity_users_cat_2020.Rdata")

###### ON THE LAPTOP ######
dir_to_read="/Users/gale/Desktop/phd/uselection2020/us_pnas_2020/alex_2016_data/data_share/"
file_to_read=list.files(dir_to_read)
all_data=data.table()
for (file in file_to_read)
{
  data=fread(paste0(dir_to_read,file))
  data=distinct(data,tweet_id,.keep_all = T)
  bias=gsub("_user_.+","",file)
  data$bias=bias
  if (grepl("non_official",file))
  {
    data$type="unofficial"
  }else
  {
    data$type="official"
  }
  all_data=rbind(all_data,data)
}

user_stat=all_data[,.(count=.N),by=user_id]
user_to_keep=user_stat[count>1,user_id]
user_to_keep=all_data[user_id %fin% user_to_keep]
user_stat=all_data[,.(count=.N),by=.(user_id, bias)]
usr_stat_unique=user_stat[,.SD[which.max(rank(count, ties.method = "random"))],by=.(user_id)]

biases=unique(usr_stat_unique$bias)

similarity_data=data.table()

for (row in biases)
{
  user=usr_stat_unique[bias==row,user_id]
  for (col in biases)
  {
    count=user_to_keep[user_id %fin% user & bias==col, .N]/user_to_keep[user_id %fin% user, .N]
    n_count=user_to_keep[user_id %fin% user & bias==col, .N]
    similarity_data=rbind(similarity_data,data.table(row=row,col=col,val=count,count=n_count))
  }
}
similarity_data_2016=similarity_data
save(similarity_data_2016, file="similarity_users_cat_2016.Rdata")
