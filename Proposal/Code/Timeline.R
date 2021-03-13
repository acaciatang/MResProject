#import data
df <- read.csv('../Data/timeline.csv')

#import packages
library(ggplot2)
library(scales)

#Data Wrangling
positions <- c(0.5, -0.5, 1.0, -1.0, 1.5, -1.5)
directions <- c(1, -1)

line_pos <- data.frame(
    "Day"=unique(df$Day),
    "position"=rep(positions, length.out=length(unique(df$Day))),
    "direction"=rep(directions, length.out=length(unique(df$Day)))
)

df <- merge(x=df, y=line_pos, by="Day", all = TRUE)
df <- df[with(df, order(Day)), ]

text_offset <- 0.12

df$month_count <- ave(df$Day==df$Day, df$Day, FUN=cumsum)
df$text_position <- (df$month_count * text_offset * df$direction) + df$position


#### PLOT ####
p<-ggplot(df,aes(x=Day,y=0, label=Task))
p<-p+labs(col="Task")
#p<-p+scale_color_manual(values=status_colors, labels=status_levels, drop = FALSE)
p<-p+theme_classic()

# Plot horizontal black line for timeline
p<-p+geom_hline(yintercept=0, 
                color = "black", size=0.3)

# Plot vertical segment lines for Tasks
p<-p+geom_segment(data=df, aes(y=position,yend=0,xend=Day), color='grey', size=0.2)

# Plot scatter points at zero and Day
p<-p+geom_point(aes(y=0, col=df$colour), size=6)

# Don't show axes, appropriately position legend
p<-p+theme(axis.line.y=element_blank(),
                 axis.text.y=element_blank(),
                 axis.title.x=element_blank(),
                 axis.title.y=element_blank(),
                 axis.ticks.y=element_blank(),
                 axis.text.x =element_blank(),
                 axis.ticks.x =element_blank(),
                 axis.line.x =element_blank(),
                 legend.position = "none"
                )

# Show text for each month
p<-p+geom_text(data=df, aes(x=Day,y=-0.15,label=Day),size=8,vjust=0.5, color='black')

# Show text for each Task
p<-p+geom_text(aes(y=text_position,label=Task),size=6)

#print as pdf
pdf("../Figures/timeline.pdf", 30, 6)
    p
graphics.off();