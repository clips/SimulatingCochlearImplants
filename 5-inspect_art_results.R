
art_dir <- paste(getwd(), sep = "/", 'results/art')

#################################################################################
# FULLY TRAINED MODELS
#################################################################################

# HIGH- vs. LOW-RES
path <- paste(art_dir, 'high_vs_lowres_gender_per.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'high_vs_lowres_gender_cnn.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'high_vs_lowres_words_per.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'high_vs_lowres_words_cnn.csv', sep = "/") 
d4 <- read.csv(path)


# MED- vs. LOW-RES, PD NETWORKS
path <- paste(art_dir, 'gender_pd_per.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_pd_cnn.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_pd_per.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_pd_cnn.csv', sep = "/") 
d4 <- read.csv(path)


# MED- vs. LOW-RES, CD NETWORKS
path <- paste(art_dir, 'gender_cd_per.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_cd_cnn.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_cd_per.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_cd_cnn.csv', sep = "/") 
d4 <- read.csv(path)


# PER vs. CNN
path <- paste(art_dir, 'per_vs_cnn_gender_pd.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'per_vs_cnn_gender_cd.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'per_vs_cnn_words_pd.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'per_vs_cnn_words_cd.csv', sep = "/") 
d4 <- read.csv(path)




#################################################################################
# MODELS TRAINED FOR 1 EPOCH
#################################################################################


# MED- vs. LOW-RES, PD NETWORKS
path <- paste(art_dir, 'gender_pd_per_1ep.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_pd_cnn_1ep.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_pd_per_1ep.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_pd_cnn_1ep.csv', sep = "/") 
d4 <- read.csv(path)


# MED- vs. LOW-RES, CD NETWORKS
path <- paste(art_dir, 'gender_cd_per_1ep.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_cd_cnn_1ep.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_cd_per_1ep.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_cd_cnn_1ep.csv', sep = "/") 
d4 <- read.csv(path)




#################################################################################
# MODELS TRAINED FOR 0 EPOCHS
#################################################################################


# MED- vs. LOW-RES, PD NETWORKS
path <- paste(art_dir, 'gender_pd_per_0ep.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_pd_cnn_0ep.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_pd_per_0ep.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_pd_cnn_0ep.csv', sep = "/") 
d4 <- read.csv(path)


# MED- vs. LOW-RES, CD NETWORKS
path <- paste(art_dir, 'gender_cd_per_0ep.csv', sep = "/") 
d1 <- read.csv(path)

path <- paste(art_dir, 'gender_cd_cnn_0ep.csv', sep = "/") 
d2 <- read.csv(path)

path <- paste(art_dir, 'words_cd_per_0ep.csv', sep = "/") 
d3 <- read.csv(path)

path <- paste(art_dir, 'words_cd_cnn_0ep.csv', sep = "/") 
d4 <- read.csv(path)