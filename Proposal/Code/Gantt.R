require(ggplot2)

activity <- c("Develop Analysis Pipeline", "Data Analysis", "Develop Analysis Pipeline", "Preparation for Experiment", "Administrate Antibiotics", "Record Bee Behaviour", "Administrate Antibiotics", "Record Bee Behaviour", "Administrate Antibiotics", "Record Bee Behaviour", "Data Analysis", "Writing")
activity <- cbind(activity, c(1:length(activity)))
start <- c("start")
end <- c("end")

Starts <- cbind(activity, start)
Ends <- cbind(activity, end)
Gantt <- as.data.frame(rbind(Starts, Ends))
Dates <- c("2020-01-01", "2020-02-15", "2020-03-01", "2020-01-01", "2020-02-02", "2020-02-01", "2020-03-02", "2020-03-01", "2020-03-18", "2020-03-17", "2020-04-01", "2020-07-01",
        "2020-02-15", "2020-02-29", "2020-03-31", "2020-01-31", "2020-02-03", "2020-02-15", "2020-03-03", "2020-03-15", "2020-03-19", "2020-03-31", "2020-06-30", "2020-07-31")
Dates <- as.Date(Dates, "%Y-%m-%d")
Gantt$date <- Dates
Gantt$activity <- factor(Gantt$activity,levels = c("Writing", "Data Analysis", "Administrate Antibiotics", "Record Bee Behaviour", "Preparation for Experiment", "Develop Analysis Pipeline"))

p <- ggplot(Gantt, aes(date, activity, color = activity, group = V2)) +
	geom_line(size = 10) +
	labs(x="Month", y=NULL)

pdf("../Figures/gnatt.pdf", 10, 3)
    p + theme(legend.position = "none") +
    scale_x_date(date_breaks = "months", date_labels = "%b")
graphics.off();
