require(ggplot2)

Gantt <- read.csv("../Data/Schedule.csv")

Gantt$date <- as.Date(Gantt$date, "%d/%m/%Y")
Gantt$activity <- factor(Gantt$activity,levels = c("Writing", "Data Analysis", "Administrate Antibiotics", "Record Bee Behaviour", "Preparation for Experiment", "Develop Analysis Pipeline"))

p <- ggplot(Gantt, aes(date, activity, color = activity, group = V2)) +
	geom_line(size = 10) +
	labs(x="Month", y=NULL)

pdf("../Figures/gnatt.pdf", 10, 3)
    p + theme(legend.position = "none") +
    scale_x_date(date_breaks = "months", date_labels = "%b")
graphics.off();