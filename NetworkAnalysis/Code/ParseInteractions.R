#import data
RawData <- read.csv("../../AnalyseBehaviours/Results/BEE_Behaviours.csv", header = T)

Edges <- RawData[!is.na(RawData$interacting), c('ID', 'interacting')]
colnames(Edges) = c("from", "to")
Nodes <- data.frame(name = unique(RawData$ID))

write.csv(Edges, '../Data/BEE_Edges.csv', row.names = FALSE)
write.csv(Nodes, '../Data/BEE_Nodes.csv', row.names = FALSE)