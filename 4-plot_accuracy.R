library(DBI)   # SQLITE
library(boot)  # bootstrapping
library(ppcor) # partial correlations
library(ggplot2)
library(extrafont)
loadfonts()

################################################################################
################################################################################


# CONNECT TO DATA BASE WITH MODEL RESULTS PER EPOCH
db_dir <- paste(getwd(), sep = "/", 'results/data_bases')
plot_dir <- paste(getwd(), sep = "/", 'results/plots')

ModelsTableName <- 'ModelsT'
epochsTableName <- 'EpochsT'
modelIDFieldName <- 'model_id'

# connect to database 
# (assumes that the working directory is the project root)
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)

# get the tables as data frames
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))

################################################################################

PlotOverEpochs <- function(modelID, numberEpochs, measure, ylim,  lwd, 
                           color='red', points=FALSE, xlab='', ylab='', 
                           main=''){
  
  query = sprintf("SELECT * from %s WHERE %s=%d", epochsTableName, 
                  modelIDFieldName, modelID)
  data = dbGetQuery(dbCon, query)
  
  y <- data[[measure]]
  y <- y[1:numberEpochs] # cuts back or pads with NAs
  epochRange = c(1:numberEpochs)
  
  # for efficiency, only plot every 10th point
  #epochRange <- epochRange[seq(1, length(epochRange), 10)]
  #y <- y[seq(1, length(y), 10)]
  
  if (points==TRUE) {
    points(epochRange, y, xlab=xlab, ylab=ylab, cex.lab=1.85, lwd=lwd, pch=19,
           cex.main=2.0, main=main, cex.axis=2.00, ylim=ylim, cex=1.0, type='l', 
           col=color)
  }
  else {
    plot(epochRange, y, xlab=xlab, ylab=ylab, cex.lab=1.85, lwd=lwd, pch=19,
         cex.main=2.0, main=main, cex.axis=2.00, ylim=ylim, cex=1.0, type='l', 
         axes=FALSE, col=color)
  }
  
}

################################################################################


# 1. HIGH-RES INPUT

axis_epochs <- c(0, 125, 250)
max_epochs <- 250

# HIGH-RES WORDS
plotpath <-  paste(plot_dir, 'words_hires.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0, paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'words_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=0, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'words_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=0, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# HIGH-RES UTTERANCES
plotpath <-  paste(plot_dir, 'gender_hires.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'gender_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=0, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'gender_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=0, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

legend('bottomright', c('PER', 'CNN'), 
       lty=1, col=c('grey70', 'black'), bty='n', cex=1.4, lwd=4)

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)

#########################################################s#######################

# 2. PD NETWORKS ON MED- AND LOW-RES INPUT

# MED-RES WORDS
plotpath <-  paste(plot_dir, 'words_pd_medres.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'words_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'words_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# LOW-RES WORDS
plotpath <-  paste(plot_dir, 'words_pd_lores.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'words_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'words_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# MED-RES GENDER
plotpath <-  paste(plot_dir, 'gender_pd_medres.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'gender_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'gender_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

legend('bottomright', c('PER', 'CNN'), 
       lty=1, col=c('grey70', 'black'), bty='n', cex=1.4, lwd=4)

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# LOW-RES GENDER
plotpath <-  paste(plot_dir, 'gender_pd_lores.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'gender_pd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'gender_pd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

legend('bottomright', c('PER', 'CNN'), 
       lty=1, col=c('grey70', 'black'), bty='n', cex=1.4, lwd=4)

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)



#################################################################################


###### CD NETWORKS ON MED- AND LOW-RES INPUT

# MED-RES WORDS
plotpath <-  paste(plot_dir, 'words_cd_medres.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'words_cd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'words_cd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# LOW-RES WORDS
plotpath <-  paste(plot_dir, 'words_cd_lores.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'words_cd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'words_cd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# MED-RES GENDER
plotpath <-  paste(plot_dir, 'gender_cd_medres.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'gender_cd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'gender_cd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=1, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

legend('bottomright', c('PER', 'CNN'), 
       lty=1, col=c('grey70', 'black'), bty='n', cex=1.4, lwd=4)

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)


# LOW-RES GENDER
plotpath <-  paste(plot_dir, 'gender_cd_lores.pdf', sep = "/") 
pdf(plotpath, width=3.5, height=3.0,paper='special', family='Times New Roman') 
par(mar=c(5.1,4.1,1.8,2.1))

dbPath <-  paste(db_dir, 'gender_cd_cnn.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='black', points=FALSE, 
               xlab='Epochs', ylab='Accuracy', main='')

dbPath <-  paste(db_dir, 'gender_cd_per.sqlite', sep = "/") 
dbCon = dbConnect(RSQLite::SQLite(), dbname=dbPath)
modelsTable = dbGetQuery(dbCon, sprintf('select * from %s', ModelsTableName))
epochsTable = dbGetQuery(dbCon, sprintf('select * from %s', epochsTableName))
PlotOverEpochs(modelID=2, numberEpochs=max_epochs, measure='accuracy_valid', 
               ylim=c(0.0, 1.0), lwd=6, color='grey70', points=TRUE, 
               xlab='Epochs', ylab='Accuracy', main='')

legend('bottomright', c('PER', 'CNN'), 
       lty=1, col=c('grey70', 'black'), bty='n', cex=1.4, lwd=4)

axis(side = 1, at = axis_epochs, cex.axis=1.85)
axis(side = 2, at = c(0.0, 0.5, 1.0), cex.axis=1.85)
box()
dev.off()
embed_fonts(plotpath)
