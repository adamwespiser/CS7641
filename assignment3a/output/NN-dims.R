library(dplyr)
library(ggplot2)
library(RColorBrewer)
ds_name = "wine-qual"
max_dims = 12
pca = read.csv(file =  paste("PCA/",ds_name,"_dim_red.csv",sep=""),
               stringsAsFactors = FALSE)
pca$dim = pca$param_pca__n_components
pca$exp = "PCA"
ica = read.csv(file =  paste("ICA/",ds_name,"_dim_red.csv",sep=""),
               stringsAsFactors = FALSE)
ica$dim = ica$param_ica__n_components
ica$exp = "ICA"
rf = read.csv(file =  paste("RF/",ds_name,"_dim_red.csv",sep=""),
               stringsAsFactors = FALSE)
rf$dim = rf$param_filter__n
rf$exp = "RF"
rf = rf %>% filter(dim <= max_dims + 1)
rp = read.csv(file =  paste("RP/",ds_name,"_dim_red.csv",sep=""),
               stringsAsFactors = FALSE)
rp$dim = rp$param_rp__n_components
rp$exp = "RP"
mc = c("dim", "exp","mean_test_score")

frame = rbind(pca[,mc], ica[,mc], rf[,mc], rp[,mc])
frame$DimRed = frame$exp
frame$exp <- NULL
frame %>%
  dplyr::group_by(dim,DimRed) %>%
  summarise(y = mean(mean_test_score)) %>%
  ggplot(aes(x = dim,
            y = y,
            color = DimRed)) +
    #geom_quantile(method = "loess") +
    geom_line(size=1) +
    ggtitle(paste(ds_name,"All NN Experiments",sep = " ")) +
    xlab("Dimensions After Reduction") +
    ylab("Mean Test Set Accuracy") + 
    theme_bw() + 
    geom_hline(yintercept = 0.72) +
    scale_colour_brewer(palette = "Set1")

ggsave(file = paste(ds_name,"DR-NN.png",sep="-"), height = 4, width = 6)


