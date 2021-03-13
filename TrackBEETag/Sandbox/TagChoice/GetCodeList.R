#load data
bkgdrm1 <- read.csv("P1000007.csv", header = T)
bkgdrm2 <- read.csv("P1000011.csv", header = T)

old1 <- read.csv("old/P1000007.csv", header = T)
old2 <- read.csv("old/P1000008.csv", header = T)
old3 <- read.csv("old/P1000010.csv", header = T)
old4 <- read.csv("old/P1000011.csv", header = T)

hamming <- read.csv("HammingDistance.csv", header = T)
taglist <- read.csv("16BitTagList.csv", header = F)

#import packages
require(tidyverse)
#install.packages('tidygraph', dependencies = TRUE)
#install.packages('ggraph', dependencies = TRUE)
require(tidygraph)
require(ggraph)

#combine datasets
bkgdrm <- rbind(bkgdrm1, bkgdrm2)
old <- rbind(old1, old2, old3, old4)
all <- rbind(bkgdrm, old)

#tabulate IDs
IDbkgdrm <- as.data.frame(sort(table(bkgdrm$ID), decreasing = TRUE))
IDold <- as.data.frame(sort(table(old$ID), decreasing = TRUE))
IDall <- as.data.frame(sort(table(all$ID), decreasing = TRUE))

newfinal <- IDbkgdrm[IDbkgdrm$Var1 %in% taglist,]
oldfinal <- IDold[IDold$Var1 %in% taglist,]
allfinal <- IDall[IDall$Var1 %in% taglist,]

newfinal <- newfinal[newfinal$Freq > 1,]
oldfinal <- oldfinal[oldfinal$Freq > 1,]
allfinal <- allfinal[allfinal$Freq > 3,]

robust <- taglist[!(taglist %in% allfinal$Var1)]

#sort into groups to maximize hamming distance
robusthamming <- hamming[hamming$X1 %in% robust,]
robusthamming <- robusthamming[robusthamming$X14 %in% robust,]

write.csv(robusthamming, "robusthamming.csv")

#visualise Hamming Distance
Nodes <- data.frame(name = as.character(unique(c(robusthamming$X1, robusthamming$X14))))
Interactions = robusthamming
Interactions$from <-  match(as.character(Interactions$X1), Nodes$name)
Interactions$to <- match(as.character(Interactions$X14), Nodes$name)
Interactions$weight <- abs(3 - Interactions$X4)
#Edges <- data.frame(from = Interactions$from, to = Interactions$to, weight = robusthamming$X4)
Edges <- data.frame(from = Interactions$from, to = Interactions$to, weight = Interactions$weight)
network <- tbl_graph(nodes = Nodes, edges = Edges)

ggraph(network, layout = "graphopt") + 
  geom_edge_link(aes(width = weight/10), colour = "#7e7e7e") +
  geom_node_point(size = 4, colour = "#ff0000") +
  geom_node_text(aes(label = name), size = 3, repel = TRUE) +
  theme_graph()

super <- robusthamming[robusthamming$X4 == 11,]
as.data.frame(sort(table(super$X1), decreasing = TRUE))

#Groups
robustlist <- read.csv("robust.csv", header = F)
robustlist$V2 <- robustlist$V2%%4
list1 <- robustlist$V1[robustlist$V2 == 1]
list2 <- robustlist$V1[robustlist$V2 == 2]
list3 <- robustlist$V1[robustlist$V2 == 3]
list4 <- robustlist$V1[robustlist$V2 == 0]

generateGham <- function(list, i){
  Gham <- robusthamming[robusthamming$X1 %in% list,]
  Gham <- Gham[Gham$X14 %in% list,]

  rmG <- Gham[Gham$X4 == 3,]
  finallist <- list[!(list %in% rmG$X14)]
  Gham2 <- robusthamming[robusthamming$X1 %in% finallist,]
  Gham2 <- Gham2[Gham2$X14 %in% list,]

  rmG2 <- Gham2[Gham2$X4 < 5,]
  listG <- rmG2$X14
  sorted <- as.data.frame(sort(table(listG), decreasing = TRUE))
  sorted$listG <- as.integer(as.character(sorted$listG))
  removeme <- sort(sorted$listG[1:(nrow(sorted)-30)])
  finallist2 <- sorted$listG[!(sorted$listG %in% removeme)]


  finalGham <- robusthamming[robusthamming$X1 %in% finallist2,]
  finalGham <- finalGham[finalGham$X14 %in% finallist2,]
  write.csv(finalGham, paste("G", i, "ham.csv", sep = ""))
  return(sort(finallist2))
}

G1 <- generateGham(list1, 1)
G2 <- generateGham(list2, 2)
G3 <- generateGham(list3, 3)
G4 <- generateGham(list4, 4)

IDlists <- cbind(G1, G2, G3, G4)
colnames(IDlists) <- c("G1", "G2", "G3", "G4")
write.csv(IDlists, "IDlists.csv")