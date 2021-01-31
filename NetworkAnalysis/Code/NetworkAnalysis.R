#import packages
require(tidyverse)
require(tidygraph)
require(ggraph)

#import data
RawData <- read.csv("../../AnalyseBehaviours/Results/BEE_Behaviours.csv", header = T)

Interactions <- RawData[!is.na(RawData$interacting), c('ID', 'interacting')]
Interactions = as.data.frame(table(Interactions))
Interactions = Interactions[Interactions$Freq > 0,]

Nodes <- data.frame(name = as.character(unique(RawData$ID)))
Edges <- data.frame(from = Interactions$ID, to = Interactions$interacting, weight = Interactions$Freq)
network <- tbl_graph(nodes = Nodes, edges = Edges)

ggraph(network, layout = "graphopt") + 
  geom_edge_link(aes(width = weight), colour = "#7e7e7e") +
  geom_node_point(size = 4, colour = "#ff0000") +
  geom_node_text(aes(label = name), size = 3, repel = TRUE) +
  theme_graph()