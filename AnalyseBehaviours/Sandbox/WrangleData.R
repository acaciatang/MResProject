#load data
RawData <- read.csv("../Data/P1000011.csv", header = T)

#load packages
require("zoo") #for rollapply
########################################################################
#correct for angle drift
Movement <- function(id){
    subset <- RawData[RawData$ID == id,]
    correction <- function(i){
        rows <- subset[(i-1):i,]
        #get distance
        distance <- sqrt((rows$centroidX[1] - rows$centroidX[2])^2 + (rows$centroidY[1] - rows$centroidY[2])^2)
        #get speed
        speed <- distance/(rows$frame[2]-rows$frame[1])
        #get angle
        movDir <- atan((rows$centroidX[1] - rows$centroidX[2])/(rows$centroidY[2] - rows$centroidY[1]))
        #get difference between angle of tag and movement
        dirDiff <- abs(movDir - rows$dir[2])
        return(c(distance, speed, movDir, dirDiff))
    }
    output <- rbind(rep(NA, 4), t(sapply(2:nrow(subset), correction)))
    colnames(output) <- c('distance', 'speed', 'movDir', 'dirDiff')
    output <- cbind(subset, output)

    write.csv(output, paste("../Results/RawMovement_", as.character(id),".csv"), row.names = F)
    return(output)
}

#output
freqID <- as.data.frame(table(RawData$ID))
freqID$Var1 <- as.numeric(as.character(freqID$Var1))
morethan2 <- freqID$Var1[freqID$Freq > 1]
RawMovements <- lapply(morethan2, Movement)

#NEED TO ADD CORRECTION!!!!! DEFINE THRESHOLD!!!!!!!

########################################################################
#for interactions
Versus <- function(frameNum) {
    subset <- RawData[RawData$frame == frameNum,]
    post <- t(combn(c(1:nrow(subset)), 2))
    #find distance between tags
    #find angle between tags
    DISTANG <- function(postRow){
        ID1 <- subset$ID[postRow[1]]
        ID2 <- subset$ID[postRow[2]]
        distance <- sqrt((subset$centroidX[postRow[1]] - subset$centroidX[postRow[2]])^2 + (subset$centroidY[postRow[1]] - subset$centroidY[postRow[2]])^2)
        angle <- abs(subset$dir[postRow[1]] - subset$dir[postRow[2]])
        angle <- abs(180 - angle)
        DistAngle = c(ID1, ID2, distance, angle)
        return(DistAngle)
    }
    
    distanceangle <- t(apply(post, 1, DISTANG))
    output <- cbind(rep(frameNum, nrow(output)), distanceangle)
    return(output)
}

#output
freq <- as.data.frame(table(RawData$frame))
freq$Var1 <- as.numeric(as.character(freq$Var1))
morethan2 <- freq$Var1[freq$Freq > 1]
RawInteractions <- mapply(Versus, morethan2, SIMPLIFY=F)
RawInteractions <- do.call(rbind, RawInteractions)
colnames(RawInteractions) = c('frame', 'Bee1', 'Bee2', 'distance', 'angle')

write.csv(RawInteractions, "../Results/RawInteractions.csv", row.names = F)
