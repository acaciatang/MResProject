Wrangled <- read.csv("../Data/BEE_Wrangled.csv", header = T)

hist(Wrangled$speed[Wrangled$speed < 10])