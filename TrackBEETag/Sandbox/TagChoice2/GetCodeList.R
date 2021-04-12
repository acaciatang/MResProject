#load data
new1 <- read.csv("1A.csv", header = T)
new2 <- read.csv("1B.csv", header = T)
new3 <- read.csv("1C.csv", header = T)
new4 <- read.csv("1D.csv", header = T)
real <- read.csv("finalIDs.csv", header = T)

hamming <- read.csv("HamDis.csv", header = T)
colnames(hamming) = c('ID1', 'ID2', "dis")
taglist <- read.csv("16BitTagList.csv", header = F)

#import packages
require(tidyverse)
#install.packages('tidygraph', dependencies = TRUE)
#install.packages('ggraph', dependencies = TRUE)
require(tidygraph)
require(ggraph)

#combine datasets
new1 = new1[new1$ID %in% real$G1,]
new2 = new2[new2$ID %in% real$G2,]
new3 = new3[new3$ID %in% real$G3,]
new4 = new4[new4$ID %in% real$G4,]
all <- rbind(new1, new2, new3, new4)

#tabulate IDs
IDall <- as.data.frame(sort(table(all$ID), decreasing = TRUE))
allfinal <- IDall[IDall$Var1 %in% taglist,]
allfinal <- allfinal[allfinal$Freq > 100,]

robust <- taglist[taglist %in% allfinal$Var1]
#write.csv(robust, "robust.csv")

#sort into groups to maximize hamming distance
robusthamming <- hamming[hamming$ID1 %in% robust,]
robusthamming <- robusthamming[robusthamming$ID2 %in% robust,]

write.csv(robusthamming, "robusthamming.csv")


#Groups
robustlist <- read.csv("robust.csv", header = F)
robustlist$V2 <- robustlist$V2%%4
list1 <- robustlist$V1[robustlist$V2 == 1]
list2 <- robustlist$V1[robustlist$V2 == 2]
list3 <- robustlist$V1[robustlist$V2 == 3]
list4 <- robustlist$V1[robustlist$V2 == 0]

generateGham <- function(list, i){
  Gham <- robusthamming[robusthamming$ID1 %in% list,]
  Gham <- Gham[Gham$ID2 %in% list,]

  rmG2 <- Gham[Gham$dis == 4,]
  listG <- rmG2$ID2
  sorted <- as.data.frame(sort(table(listG), decreasing = TRUE))
  sorted$listG <- as.integer(as.character(sorted$listG))
  removeme <- sort(sorted$listG[1:(nrow(sorted)-40)])
  finallist2 <- sorted$listG[!(sorted$listG %in% removeme)]

  finalGham <- robusthamming[robusthamming$ID1 %in% finallist2,]
  finalGham <- finalGham[finalGham$ID2 %in% finallist2,]
  write.csv(finalGham, paste("G", i, "ham.csv", sep = ""))
  return(sort(finallist2))
}

G1 <- generateGham(robustlist, 0)
G2 <- generateGham(list2, 2)
G3 <- generateGham(list3, 3)
G4 <- generateGham(list4, 4)

IDlists <- cbind(G1, G2, G3, G4)
colnames(IDlists) <- c("G1", "G2", "G3", "G4")
write.csv(IDlists, "IDlists.csv")

write.csv(sort(finallist2), 'drawme.csv')