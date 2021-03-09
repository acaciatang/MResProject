#load data
bkgdrm1 <- read.csv("P1000007.csv", header = T)
bkgdrm2 <- read.csv("P1000011.csv", header = T)

old1 <- read.csv("old/P1000007.csv", header = T)
old2 <- read.csv("old/P1000008.csv", header = T)
old3 <- read.csv("old/P1000010.csv", header = T)
old4 <- read.csv("old/P1000011.csv", header = T)

hamming <- read.csv("HammingDistance.csv", header = T)
taglist <- read.csv("16BitTagList.csv", header = F)

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