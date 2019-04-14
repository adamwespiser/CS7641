library(ggplot2)

merge_q_files <- function(epfile, ffile){
  dfe = read.csv(file = epfile, stringsAsFactors = FALSE)
  df = read.csv(file = ffile, stringsAsFactors = FALSE)
  dm = merge(dfe, df, by.x="episode", by.y="steps")
  return(dm)
}
output_dir = "~/sandbox/output-v4-all"
output_dir = "output"
#Cliff
c_ep = paste(output_dir, "/report/Q/cliff_walking_0.5_random_0.5_0.0001_0.7_episode.csv", sep = "")
c_csv = paste(output_dir, "/report/Q/cliff_walking_0.5_random_0.5_0.0001_0.7.csv", sep = "")
c_merge = merge_q_files(c_ep, c_csv)
#Frozen Lake
#
f_ep = paste(output_dir, "/report/Q/frozen_lake_20x20_v6_0.5_0_0.3_0.0001_0.53_episode.csv", sep = "")
f_csv = paste(output_dir, "/report/Q/frozen_lake_20x20_v6_0.5_0_0.3_0.0001_0.53.csv", sep = "")
f_merge = merge_q_files(f_ep, f_csv)
library(ggplot2)
plotit <- function(mdf, title, outfile){
   ggplot(mdf, aes(x=reward.y)) + geom_histogram(bins=30) + 
     xlab("Episodic Reward") + 
     ylab("Frequency")+
     ggtitle(title) + 
     geom_vline(aes(xintercept=mean(mdf$reward.y)),color='red') + 
     theme_bw()
    ggsave(outfile, height=4, width=4)
}
plotit2 <- function(mdf, title, outfile){
   ggplot(mdf, aes(x=length)) + geom_histogram(bins=30) + 
     xlab("Steps Per Episode") + 
     ylab("Frequency")+
     ggtitle(title) + 
     geom_vline(aes(xintercept=mean(mdf$length)),color='red') + 
     theme_bw()
    ggsave(outfile, height=4, width=4)
}

plotit(c_merge, "Cliff - Q Learning \nEpisode Reward Histogram", "cliff_q_epi_histogram.png")
plotit(f_merge, "Frozen Lake - Q Learning \nEpisode Reward Histogram", "frozen_q_epi_histogram.png")

plotit2(c_merge, "Cliff - Q Learning \nEpisode Length Histogram", "cliff_q_len_histogram.png")
plotit2(f_merge, "Frozen Lake - Q Learning \nEpisode Length Histogram", "frozen_q_len_histogram.png")


glm(time.y ~ lenth, family="gaussian", data = c_merge)
glm(time.y ~ lenth, family="gaussian", data = f_merge)


