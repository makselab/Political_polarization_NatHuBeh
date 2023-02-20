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
library(igraph)
library(dplyr)
library(reshape2)
library(Matrix)
#### CA ALGORITHM (BARBERA) ####
CA <- function (obj, nd = NA, suprow = NA, supcol = NA, subsetrow = NA,
                subsetcol = NA, verbose=TRUE)
{
  if (verbose) message("Preparing matrix object...")
  nd0 <- nd
  I <- dim(obj)[1]
  J <- dim(obj)[2]
  rn <- dimnames(obj)[[1]]
  cn <- dimnames(obj)[[2]]
  N <- matrix(as.matrix(obj), nrow = I, ncol = J)
  Ntemp <- N
  NtempC <- NtempR <- N
  rm("N")
  suprow <- sort(suprow)
  supcol <- sort(supcol)
  if (!is.na(supcol[1]) & !is.na(suprow[1])) {
    NtempC <- Ntemp[-suprow, ]
    NtempR <- Ntemp[, -supcol]
  }
  if (!is.na(supcol[1])) {
    SC <- as.matrix(NtempC[, supcol])
    Ntemp <- Ntemp[, -supcol]
    cs.sum <- apply(SC, 2, sum)
  }
  rm("NtempC")
  if (!is.na(suprow[1])) {
    SR <- matrix(as.matrix(NtempR[suprow, ]), nrow = length(suprow))
    Ntemp <- Ntemp[-suprow, ]
    rs.sum <- apply(SR, 1, sum)
  }
  rm("NtempR")
  N <- matrix(as.matrix(Ntemp), nrow = dim(Ntemp)[1], ncol = dim(Ntemp)[2])
  subsetrowt <- subsetrow
  if (!is.na(subsetrow[1]) & !is.na(suprow[1])) {
    subsetrowi <- subsetrow
    subsetrowt <- sort(c(subsetrow, suprow))
    subsetrowt <- subsetrowt[!duplicated(subsetrowt)]
    I <- length(subsetrowt)
    for (q in length(suprow):1) {
      subsetrow <- subsetrow[subsetrow != suprow[q]]
      subsetrow <- subsetrow - as.numeric(suprow[q] < subsetrow)
    }
    for (q in 1:length(suprow)) suprow[q] <- (1:length(subsetrowt))[subsetrowt ==
                                                                      suprow[q]]
  }
  subsetcolt <- subsetcol
  if (!is.na(subsetcol[1]) & !is.na(supcol[1])) {
    subsetcoli <- subsetcol
    subsetcolt <- sort(c(subsetcol, supcol))
    subsetcolt <- subsetcolt[!duplicated(subsetcolt)]
    J <- length(subsetcolt)
    for (q in length(supcol):1) {
      subsetcol <- subsetcol[subsetcol != supcol[q]]
      subsetcol <- subsetcol - as.numeric(supcol[q] < subsetcol)
    }
    for (q in 1:length(supcol)) supcol[q] <- (1:length(subsetcolt))[subsetcolt ==
                                                                      supcol[q]]
  }
  dim.N <- dim(N)
  if (!is.na(subsetrow[1])) {
    if (!is.na(supcol[1]))
      SC <- as.matrix(SC[subsetrow, ])
  }
  if (!is.na(subsetcol[1])) {
    if (!is.na(suprow[1]))
      SR <- matrix(as.matrix(SR[, subsetcol]), nrow = length(suprow))
  }
  if (is.na(subsetrow[1]) & is.na(subsetcol[1])) {
    nd.max <- min(dim.N) - 1
  }
  else {
    N00 <- N
    if (!is.na(subsetrow[1]))
      N00 <- N00[subsetrow, ]
    if (!is.na(subsetcol[1]))
      N00 <- N00[, subsetcol]
    dim.N <- dim(N00)
    nd.max <- min(dim.N)
    if (!is.na(subsetrow[1]) & is.na(subsetcol[1])) {
      if (dim.N[1] > dim.N[2])
        nd.max <- min(dim.N) - 1
    }
    else {
      if (is.na(subsetrow[1]) & !is.na(subsetcol[1])) {
        if (dim.N[2] > dim.N[1]) {
          nd.max <- min(dim.N) - 1
        }
      }
    }
  }
  if (verbose) message("Standardizing matrix...")
  if (is.na(nd) | nd > nd.max)
    nd <- nd.max
  n <- sum(N)
  P <- N/n
  rm <- apply(P, 1, sum)
  cm <- apply(P, 2, sum)
  eP <- rm %*% t(cm)
  S <- (P - eP)/sqrt(eP)
  rm("eP")
  rm("P")
  if (!is.na(subsetcol[1])) {
    S <- S[, subsetcol]
    cm <- cm[subsetcol]
    cn <- cn[subsetcolt]
  }
  if (!is.na(subsetrow[1])) {
    S <- S[subsetrow, ]
    rm <- rm[subsetrow]
    rn <- rn[subsetrowt]
  }
  #chimat <- S^2 * n
  if (verbose) message("Computing SVD...")
  dec <- svd(S)
  sv <- dec$d[1:nd.max]
  u <- dec$u
  v <- dec$v
  ev <- sv^2
  cumev <- cumsum(ev)
  totin <- sum(ev)
  rin <- apply(S^2, 1, sum)
  cin <- apply(S^2, 2, sum)
  rm("S")
  rm("dec")
  rachidist <- sqrt(rin/rm)
  cachidist <- sqrt(cin/cm)
  rchidist <- rep(NA, I)
  cchidist <- rep(NA, J)
  if (!is.na(subsetrow[1])) {
    obj <- obj[subsetrowt, ]
  }
  if (!is.na(subsetcol[1])) {
    obj <- obj[, subsetcolt]
  }
  ###
  if (!is.na(suprow[1])) {
    if (is.na(supcol[1])) {
      P.stemp <- matrix(as.matrix(obj[suprow, ]), nrow = length(suprow))
    }
    else {
      P.stemp <- matrix(as.matrix(obj[suprow, -supcol]),
                        nrow = length(suprow))
    }
    P.stemp <- P.stemp/apply(P.stemp, 1, sum)
    P.stemp <- t((t(P.stemp) - cm)/sqrt(cm))
    rschidist <- sqrt(apply(P.stemp^2, 1, sum))
    rchidist[-suprow] <- rachidist
    rchidist[suprow] <- rschidist
    rm("P.stemp")
  }
  else rchidist <- rachidist
  if (!is.na(supcol[1])) {
    if (is.na(suprow[1])) {
      P.stemp <- as.matrix(obj[, supcol])
    }
    else P.stemp <- as.matrix(obj[-suprow, supcol])
    P.stemp <- t(t(P.stemp)/apply(P.stemp, 2, sum))
    P.stemp <- (P.stemp - rm)/sqrt(rm)
    cschidist <- sqrt(apply(P.stemp^2, 2, sum))
    cchidist[-supcol] <- cachidist
    cchidist[supcol] <- cschidist
    rm("P.stemp")
  }
  else cchidist <- cachidist
  phi <- as.matrix(u[, 1:nd])/sqrt(rm)
  gam <- as.matrix(v[, 1:nd])/sqrt(cm)
  if (verbose) message("Projecting rows...")
  if (!is.na(suprow[1])) {
    cs <- cm
    gam.00 <- gam
    base2 <- SR/matrix(rs.sum, nrow = nrow(SR), ncol = ncol(SR))
    base2 <- t(base2)
    cs.0 <- matrix(cs, nrow = nrow(base2), ncol = ncol(base2))
    svphi <- matrix(sv[1:nd], nrow = length(suprow), ncol = nd,
                    byrow = TRUE)
    base2 <- base2 - cs.0
    phi2 <- (t(as.matrix(base2)) %*% gam.00)/svphi
    phi3 <- matrix(NA, ncol = nd, nrow = I)
    phi3[suprow, ] <- phi2
    phi3[-suprow, ] <- phi
    rm0 <- rep(NA, I)
    rm0[-suprow] <- rm
    P.star <- SR/n
    rm0[suprow] <- NA
    rin0 <- rep(NA, I)
    rin0[-suprow] <- rin
    rin <- rin0
  }
  if (verbose) message("Projecting columns...")
  if (!is.na(supcol[1])) {
    rs <- rm
    phi.00 <- phi
    base2 <- SC/matrix(cs.sum, nrow = nrow(SC), ncol = ncol(SC),
                       byrow = TRUE)
    rs.0 <- matrix(rs, nrow = nrow(base2), ncol = ncol(base2))
    svgam <- matrix(sv[1:nd], nrow = length(supcol), ncol = nd,
                    byrow = TRUE)
    base2 <- base2 - rs.0
    gam2 <- (as.matrix(t(base2)) %*% phi.00)/svgam
    gam3 <- matrix(NA, ncol = nd, nrow = J)
    gam3[supcol, ] <- gam2
    gam3[-supcol, ] <- gam
    cm0 <- rep(NA, J)
    cm0[-supcol] <- cm
    P.star <- SC/n
    cm0[supcol] <- NA
    cin0 <- rep(NA, J)
    cin0[-supcol] <- cin
    cin <- cin0
  }
  if (exists("phi3"))
    phi <- phi3
  if (exists("gam3"))
    gam <- gam3
  if (exists("rm0"))
    rm <- rm0
  if (exists("cm0"))
    cm <- cm0
  ca.output <- list(sv = sv, nd = nd0, rownames = rn, rowmass = rm,
                    rowdist = rchidist, rowinertia = rin, rowcoord = phi,
                    rowsup = suprow, colnames = cn, colmass = cm, coldist = cchidist,
                    colinertia = cin, colcoord = gam, colsup = supcol, call = match.call())
  class(ca.output) <- "ca"
  if (verbose) message("Done!")
  return(ca.output)
}
##### GENERATE IDEOLOGY FOR VARIOUS THRESHOLDS ######
############### PARAMETERS: WEIGHT THRESHOLD, NUMBER OF CONNECTIONS ##########
####################################################
influencers=fread("top_100_influencers_2020.csv")
influencers=as.character(influencers$id)
data=fread("ideology_retweet_network.csv", header=T)
data$retweet_id=NULL
colnames(data)=c("tweet_id","author_uid","retweeted_uid")
edge_list=data
edge_list$tweet_id=NULL
###### subsample ######
#edge_list=edge_list[sample(nrow(edge_list), 43133092)]
########################################
edge_list=as.matrix(edge_list[,.(as.character(author_uid),as.character(retweeted_uid))])
g=graph.edgelist(edge_list,directed = T)
E(g)$weight=1
g=igraph::simplify(g,edge.attr.comb = "sum")
##### IF WE WANT LOG WEIGHTS #####
#E(g)$weight=log(E(g)$weight)
#g=subgraph.edges(g,E(g)[weight>0], delete.vertices = T)
#### IF YOU WANT TO EXCLUDE SINGLE RETWEETS ####
#g_subgraph=subgraph.edges(g,E(g)[weight>(1)], delete.vertices = T)
################################################

####### GET WEIGHTED MATRIX ########
interaction_matrix=get.adjacency(g,type="upper",attr = "weight", sparse = igraph_opt("sparsematrices"))

#### ONLY INFLUENCERS ON COLUMNS ###
interaction_matrix=interaction_matrix[!rownames(interaction_matrix) %in% influencers,]
interaction_matrix=interaction_matrix[,colnames(interaction_matrix) %in% influencers]

#### KEEP ONLY USERS THAT HAVE MORE THAN 3 LINKS ####
#interaction_matrix=as.matrix(interaction_matrix)
rs = rowSums(interaction_matrix>0)
interaction_matrix=interaction_matrix[rs>=3,]

### EXCLUDE all 0 columns
cs= colSums(interaction_matrix>0)

#### WHO ARE THE MISSING INFLUENCERS?
print(colnames(interaction_matrix[,cs<2]))

### CONSIDER ONLY COLUMNS WITH MORE THAN ONE USER ###
interaction_matrix=interaction_matrix[,cs>1]

### CONSIDER ONLY ROW WITH AT LEAST ONE INFLUENCER NOW ###
rs = rowSums(interaction_matrix>0)
interaction_matrix=interaction_matrix[rs>0,]
## APPLY CA ALGORITHM ###
res = CA(interaction_matrix, nd=3)

### SAVE RESULTS ###
save(res, file="CA_weighted_analysis_2020_n_link_3_reduced_infl.Rdata")
save(interaction_matrix,file="weighted_interaction_matrix_2020_3_reduced_infl.Rdata")

