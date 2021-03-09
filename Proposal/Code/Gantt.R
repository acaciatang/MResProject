require(ggplot2)

#Gantt <- read.csv("../Data/Schedule.csv")
Gantt <- read.csv("../Data/DailySchedule.csv")

#Gantt$date <- as.Date(Gantt$date, "%d/%m/%Y")
Gantt$time <- as.POSIXct(Gantt$time,format="%H:%M")
Gantt$activity <- factor(Gantt$activity,levels = rev(unique(Gantt$activity)))
Gantt$V2 <- factor(Gantt$V2,levels = rev(unique(Gantt$V2)))

#p <- ggplot(Gantt, aes(date, activity, color = activity, group = V2)) +
#	geom_line(size = 6) +
#	labs(x="Month", y=NULL)

p <- ggplot(Gantt, aes(time, V2, color = V2, group = activity)) +
	geom_line(size = 6) +
	labs(x="Time", y=NULL)

pdf("../Figures/dailygantt.pdf", 10, 4)
#png("../Figures/dailygantt.png", 5000, 2000)
    p + theme(legend.position = "none") #+
    #scale_x_date(date_breaks = "months", date_labels = "%b")
graphics.off();