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


library(data.table)
library(ggplot2)
library(ggpubr)
library(latex2exp)
library(xtable)
library(latex2exp)
library(dplyr)
library(ggpattern)
library(cowplot)
library(viridis)
library(igraph)
color_values=c("#282828","#8F100B","#DB4742","#CFDB00","#4495DB","#082E4F")
load("Desktop/phd/uselection2020/us_pnas_2020/table_data_unique_users_2016.Rdata")
load("Desktop/phd/uselection2020/us_pnas_2020/table_data_unique_users_2020.Rdata")


n_2020_unique_tw=72738094
n_2020_unique_usr=sum(table_data_for_plot_2020$N_u)
n_2016_unique_tw=30756752
n_2016_unique_usr=sum(table_data_for_plot_2016$N_u)

data_old_all=table_data_for_plot_2016
data_old_all$bias=as.character(data_old_all$bias)
data_old_all$bias[data_old_all$bias=="Fake news"]="Fake news & \nextreme bias"
data_old_all$bias[data_old_all$bias=="Extreme bias right"]="Fake news & \nextreme bias"
data_old_all$bias[data_old_all$bias=="Extreme bias left"]="Fake news & \nextreme bias"

data_old_all_plot=data_old_all[,.(n_tweet=sum(N_t),n_user=sum(N_u)), by=.(bias)]
data_old_all_plot$bias=factor(data_old_all_plot$bias, 
                              levels=rev(c("Fake news & \nextreme bias",
                                       "Right news",
                                       "Right leaning news",
                                       "Center news",
                                       "Left leaning news",
                                       "Left news")),
                              labels = rev(c("Fake news & \nextreme bias",
                                         "Right",
                                         "Right \nleaning",
                                         "Center",
                                         "Left \nleaning",
                                         "Left")))
data_old_all_plot$year="2016"

#### NEW DATA ####
data_new_all=table_data_for_plot_2020
data_new_all$bias=as.character(data_new_all$bias)
data_new_all$bias[data_new_all$bias=="Fake news"]="Fake news & \nextreme bias"
data_new_all$bias[data_new_all$bias=="Extreme bias right"]="Fake news & \nextreme bias"
data_new_all$bias[data_new_all$bias=="Extreme bias left"]="Fake news & \nextreme bias"

data_new_all_plot=data_new_all[,.(n_tweet=sum(N_t),n_user=sum(N_u)), by=.(bias)]
data_new_all_plot$bias=factor(data_new_all_plot$bias, 
                              levels=rev(c("Fake news & \nextreme bias",
                                       "Right news",
                                       "Right leaning news",
                                       "Center news",
                                       "Left leaning news",
                                       "Left news")),
                              labels = rev(c("Fake news & \nextreme bias",
                                         "Right",
                                         "Right \nleaning",
                                         "Center",
                                         "Left \nleaning",
                                         "Left")))
data_new_all_plot$year="2020"

data_new_all_plot$n_tweet=data_new_all_plot$n_tweet/n_2020_unique_tw
data_new_all_plot$n_user=data_new_all_plot$n_user/n_2020_unique_usr
data_old_all_plot$n_tweet=data_old_all_plot$n_tweet/n_2016_unique_tw
data_old_all_plot$n_user=data_old_all_plot$n_user/n_2016_unique_usr

barplot_stat=rbind(data_new_all_plot,data_old_all_plot)

data_all_together=rbind(barplot_stat[,.(bias,count=n_tweet,year,type="Proportion of tweets")],
                        barplot_stat[,.(bias,count=n_user,year,type="Proportion of users")])

#### PATTERN ####
color_stripes_values=c("#6F6F6F","#D00E0E","#FF6B66","#A3AC00","#3572A7","#1362A6")
all_bar=ggplot(data_all_together, aes(x=bias,y=count, fill=bias, group=year, color=bias))+
  geom_col_pattern(position = position_dodge(width=0.9), width = 0.8,
                   pattern =c("stripe","stripe","stripe","stripe","stripe","stripe","none","none","none","none","none","none",
                              "stripe","stripe","stripe","stripe","stripe","stripe","none","none","none","none","none","none"),
                   pattern_angle = 45,
                   pattern_density = .1,
                   pattern_spacing = .04,
                   pattern_fill = rep(color_stripes_values,4),
                   pattern_colour=rep(color_stripes_values,4))+
  scale_fill_manual(values =rev(color_values))+
  scale_color_manual(values =rev(color_values))+
  theme_minimal()+
  theme(legend.position = "bottom",
        text=element_text(size=30),
        axis.text.x=element_text(angle=60, hjust=1, size=26),
        axis.title.x=element_blank())+
  facet_grid(.~type, switch = 'y')+
  theme(axis.title.y = element_blank(),
        strip.placement = "outside",
        panel.spacing = unit(2, "lines"),
        plot.margin = unit(c(1,1,1,1), "lines"))+
  guides(group = guide_legend(override.aes = 
                                list(
                                  pattern = c("none", "stripe"),
                                  pattern_spacing = .01,
                                  pattern_angle = c(0, 0, 0, 45)
                                )),
                              colour =  FALSE,
                              fill = FALSE,
                              pattern=T
  )+
  coord_fixed(ratio = 8/1)


load("Desktop/phd/uselection2020/us_pnas_2020/similarity_users_cat_2016.Rdata")
load("Desktop/phd/uselection2020/us_pnas_2020/similarity_users_cat_2020.Rdata")
similarity_data$row=factor(similarity_data$row, levels = rev(c("Fake news","Extreme bias right", "Right news", "Right leaning news",
                                                                 "Center news", "Left leaning news", "Left news","Extreme bias left")),
                              labels= rev(c("Fake news","Extreme bias\nright", "Right news", "Right leaning\nnews",
                                        "Center news", "Left leaning\nnews", "Left news","Extreme bias\nleft")))
similarity_data$col=factor(similarity_data$col, levels = rev(c("Fake news","Extreme bias right", "Right news", "Right leaning news",
                                                                 "Center news", "Left leaning news", "Left news","Extreme bias left")),
                              labels= rev(c("Fake news","Extreme bias\nright", "Right news", "Right leaning\nnews",
                                        "Center news", "Left leaning\nnews", "Left news","Extreme bias\nleft")))

similarity_data_2016$row=factor(similarity_data_2016$row, levels =  rev(c("fake","far_right", "right", "lean_right",
                                                                      "center", "lean_left", "left","far_left")),
                           labels= rev(c("Fake news","Extreme bias\nright", "Right news", "Right leaning\nnews",
                                     "Center news", "Left leaning\nnews", "Left news","Extreme bias\nleft")))
similarity_data_2016$col=factor(similarity_data_2016$col, levels = rev(c("fake","far_right", "right", "lean_right",
                                                                     "center", "lean_left", "left","far_left")),
                           labels= rev(c("Fake news","Extreme bias\nright", "Right news", "Right leaning\nnews",
                                     "Center news", "Left leaning\nnews", "Left news","Extreme bias\nleft")))

all_similarity_data=rbind(similarity_data_2016[,.(row,col,val,count,year="2016")],
                          similarity_data[,.(row,col,val,count,year="2020")])

all_tile=ggplot(all_similarity_data, aes(x=col, y=row,fill=val))+
            geom_tile()+
            scale_fill_viridis(limits=c(0,0.7),#breaks = c(0, 0.2, 0.4, 0.6),
                               guide = guide_colourbar(draw.ulim = T, draw.llim = T,
                                                       barwidth = 1, barheight =24))+
            theme_classic()+
            theme(text=element_text(size=30),
                  #axis.title = element_blank(),
                  axis.line = element_blank(),
                  axis.text = element_text(size=28),
                  axis.text.x = element_text(angle=90, hjust=1, vjust=0.5),
                  panel.spacing = unit(2, "lines"),
                  strip.background = element_rect(colour="white", fill="white"),
                  strip.text.x=element_text(size=30),
                  plot.margin = unit(c(1,1,1,1), "lines"))+
            labs(x="Links category ", y="User main category", fill="Proportion\nof links")+
            coord_fixed()+
            facet_grid(.~year)

#all_tile
ggarrange(all_bar,all_tile, nrow=1)

all_data_in_one_table=rbind(table_data_for_plot_2016,table_data_for_plot_2020)
print(xtable(all_data_in_one_table, type = "latex"))

##### FAKE PLOT FOR LEGEND ####
fake_stat=barplot_stat[,.(y=sum(n_tweet)), by=year]

ggplot(fake_stat, aes(year,y, color=year)) +    # Modify colors of pattern
  geom_col_pattern(
                   pattern =c("stripe","none"),
                   pattern_angle = 45,
                   pattern_density = .1,
                   pattern_spacing = .04,
               pattern_color = "black",
               pattern_fill = "black")+
  scale_color_grey()+
  theme(legend.direction = "horizontal",
        text=element_text(size=22))+
  guides(color = guide_legend(override.aes = 
                               list(
                                 pattern = c("none", "stripe"),
                                 pattern_spacing = .04,
                                 pattern_density = .1,
                                 pattern_angle = c(0,45)
                               )
  ))


####### NETWORK ON MATRICES #####
adj_2016=dcast(all_similarity_data[year=="2016",.(from=row,to=col,weight=val)],from~to, value.var = "weight")
adj_2016=as.matrix(adj_2016[,2:ncol(adj_2016)])

rownames(adj_2016)=colnames(adj_2016)

adj_2016=tcrossprod(adj_2016)
#mode(adj_2016) <- "numeric"
#g_2016=graph.data.frame(all_similarity_data[year=="2016",.(from=row,to=col,weight=val)])
g_2016=graph_from_adjacency_matrix(adj_2016, mode="undirected", weighted = T, diag = T)
cl_2016=cluster_louvain(as.undirected(g_2016))
V(g_2016)$membership=cl_2016$membership

plot(g_2016, layout=layout.fruchterman.reingold(g_2016),
     vertex.size=12,
     vertex.color=V(g_2016)$membership, 
     vertex.label.cex=0.7,
     vertex.label.color="black",
     vertex.frame.color="transparent",
     edge.arrow.size=0
)


### 2020 ###
adj_2020=dcast(all_similarity_data[year=="2020",.(from=row,to=col,weight=val)],from~to, value.var = "weight")
adj_2020=as.matrix(adj_2020[,2:ncol(adj_2020)])

rownames(adj_2020)=colnames(adj_2020)

adj_2020=tcrossprod(adj_2020)
#mode(adj_2016) <- "numeric"
#g_2016=graph.data.frame(all_similarity_data[year=="2016",.(from=row,to=col,weight=val)])
g_2020=graph_from_adjacency_matrix(adj_2020, mode="undirected", weighted = T, diag = T)
cl_2020=cluster_louvain(g_2020)
V(g_2020)$membership=cl_2020$membership

plot(g_2020, layout=layout.fruchterman.reingold(g_2020),
     vertex.size=12,
     vertex.color=V(g_2016)$membership, 
     vertex.label.cex=0.7,
     vertex.label.color="black",
     vertex.frame.color="transparent",
     edge.arrow.size=0
)


cl_2020$modularity
cl_2016$modularity

