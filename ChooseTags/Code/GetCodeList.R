#load data
bkgdrm1 <- read.csv("../Data/P1000007.csv", header = T)
bkgdrm2 <- read.csv("../Data/P1000011.csv", header = T)

old1 <- read.csv("../Data/old/P1000007.csv", header = T)
old2 <- read.csv("../Data/old/P1000008.csv", header = T)
old3 <- read.csv("../Data/old/P1000010.csv", header = T)
old4 <- read.csv("../Data/old/P1000011.csv", header = T)

hamming <- read.csv("../Data/HamDis.csv", header = F)
taglist <- read.csv("../Data/16BitTagList.csv", header = F)

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
#IDbkgdrm <- as.data.frame(sort(table(bkgdrm$ID), decreasing = TRUE))
#IDold <- as.data.frame(sort(table(old$ID), decreasing = TRUE))
IDall <- as.data.frame(sort(table(all$ID), decreasing = TRUE))

#newfinal <- IDbkgdrm[IDbkgdrm$Var1 %in% taglist,]
#oldfinal <- IDold[IDold$Var1 %in% taglist,]
allfinal <- IDall[IDall$Var1 %in% taglist,]

#newfinal <- newfinal[newfinal$Freq > 1,]
#oldfinal <- oldfinal[oldfinal$Freq > 1,]
allfinal <- allfinal[allfinal$Freq > 3,]

robust <- taglist[!(taglist %in% allfinal$Var1)]
write.csv(robust, "../Data/robust.csv")
#sort into groups to maximize hamming distance
robusthamming <- hamming[hamming$V1 %in% robust,]
robusthamming <- robusthamming[robusthamming$V2 %in% robust,]

write.csv(robusthamming, "../Data/robusthamming.csv")

Gham <- robusthamming[robusthamming$V1 %in% robust,]
Gham <- Gham[Gham$V2 %in% robust,]

rmG <- Gham[Gham$V3 < 4,]
rmlist <- rmG$V2
sorted <- as.data.frame(sort(table(rmlist), decreasing = TRUE))
sorted$rmlist <- as.integer(as.character(sorted$rmlist))
removeme <- sort(sorted$rmlist[1:(nrow(sorted)-50)])
finallist <- sorted$rmlist[!(sorted$rmlist %in% removeme)]

write.csv(sort(finallist), "../Results/IDlist.csv")

finalGham <- robusthamming[robusthamming$V1 %in% finallist,]
finalGham <- finalGham[finalGham$V2 %in% finallist,]
write.csv(finalGham, paste("../Results/TagsHamDis.csv", sep = ""))
