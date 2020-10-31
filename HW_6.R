

####################################################
library(rpart)
library(tm)  ## Text Mining Package
#install.packages("tm")
library(stringr)  #String manipulation
library(wordcloud)  #Vis
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)  ## For ARM
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
#install.packages("Epi")
library(Epi)
#install.packages("e1071")
library(e1071)
#install.packages("mlr")
#install.packages("rlang")
library(rlang)
library(ggplot2)
#install.packages("stringr")
library(stringr)
#install.packages("e1071")
library(e1071)
#install.packages("mlr")
library(mlr)
#install.packages("caret")
library(caret)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("e1071")
library(e1071)
#install.packages("mlr")
library(mlr)
# install.packages("caret")
library(caret)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("mclust")
library(mclust)
#install.packages("cluster")
library(cluster)
#install.packages("tm")
library(tm)
## install.packages("rpart")
#install.packages('rattle')
#install.packages('rpart.plot')
## install.packages('RColorBrewer')
# install.packages("Cairo")
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
# install.packages("philentropy")
library(philentropy)
#install.packages("forcats")
library(forcats)
# install.packages("lsa")
library(lsa) #for cosine similarity
# install.packages("igraph")
library(igraph)  #to create network of cos sim matrix
# install.packages("ggplot2")
#install.packages("rlang")
library(ggplot2)
#install.packages("corrplot")
library(corrplot)
#install.packages("pastecs") ## for stats
library(pastecs)
##install.packages("dplyr")
library(dplyr)
#install.packages("cowplot")
#library(cowplot)
#install.packages("ggpubr")
library(ggpubr)
#install.packages("english")
library(english)
library(class) #for knn
library(lattice)
#################for example data (titanic):
#install.packages("FSelector")
#install.packages("data.tree")
#install.packages("caTools")
library(FSelector)
library(data.tree)
library(caTools)
#install.packages("ElemStatLearn")
library(ElemStatLearn)
library(randomForest)
#install.packages("GGally")
#install.packages("gmodels")
library(GGally)
library(gmodels)
library(igraph)  #to create network of cos sim matrix
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
# install.packages("philentropy")
library(philentropy)
# install.packages("forcats")
library(forcats)
# install.packages("lsa")
library(lsa) #for cosine similarity

setwd("C:/Users/andy_white/Desktop/Projects/Syracuse/IST 707/R/IST_707")

pixel_real_test <- read.csv(file = "test.csv",header = T)
pixel_train <- read.csv(file = "Kaggle-train1400.csv",header = T,sep = ",")
str(pixel_train)
pixel_train2 <- pixel_train
pixel_train <- pixel_train2
#Naive bayes works with numerical data
pixel_train <- pixel_train %>%
  mutate_all(as.numeric)

#it is likely we may have to convert zeros to N/A and utilize na.rm
#We see there are approximately 100 non zero values vs 600+ zero values
which(t(pixel_train[1,]!=0))

#Might need to label data as categorical factor, but will test later
pixel_train$label <- factor(pixel_train$label)
str(pixel_train)



#lets try to sequence and select a smaller sample of the train
#we tried sampling every 5th for the test set, but didn't work out so well, lower the sample
LL <- c( 1:nrow(pixel_train))
all_sample_vec <- LL[seq(1,length(LL),15)]

#all_sample_test <-LL[seq(3,length(LL),1000)]
#we will try a sample size that is 10th the actual training size to see if that will yeild interesting results
pixel_train_v2 <- pixel_train[-c(all_sample_vec),]
pixel_test_v2 <- pixel_train[c(all_sample_vec),]
str(pixel_train_10)  # we have 4200 rows now, let's test with naive bayes classifier

#Naive bayes classifier with na.pass:
NB_pixel_train10 <- naiveBayes(label~.,data = pixel_train_v2,na.action = na.pass)
#NB_fit1 <- naive_bayes(pixel_train_10$pixel_label~.,data = pixel_train_10)

#prepare test data (remove labels)
pixel_test_10_label <- pixel_test_v2$label
#pixel_test_10_label_v2 <- pixel_test_10_label[1:15]

#first take the first 10 rows
pixel_test_10 <- pixel_test_v2[,-1]
#pixel_test_10 <- unlist(pixel_test_10)


#Now predict using prediction fromula
NB_pixel_train10_prediction <- predict(NB_pixel_train10,pixel_test_10)
#NBpred1 <- predict(NB_pixel_train10,as.data.frame(pixel_test_10))
#NB_pixel_train10

table(NB_pixel_train10_prediction,pixel_test_10_label)
confusionMatrix(NB_pixel_train10_prediction,pixel_test_10_label)
data.frame(NB_pixel_train10_prediction,pixel_test_10_label)

##### What happens if we make all 0's into NA's - will it improve the results?
#next lets make all 0's into na's to run the code a little more efficiently
pixel_label <- pixel_train$label #first separate out the label - as there are 0's that need to be accounted for
pixel_train <- pixel_train[,2:ncol(pixel_train)]
pixel_train[pixel_train==0] <- NA
pixel_train <- cbind(pixel_label,pixel_train)

#testing to see if any rows have errors or na
df_row <-row_sums(pixel_train[,2:ncol(pixel_train)],na.rm = T)
summary(df_row)
df_col <- col_sums(pixel_train2,na.rm = T)

treefit_pixel <- rpart(pixel_train_v2$label ~.,data = pixel_train_v2,method = 'class')

summary(treefit)
#can't really see trying a new packaage
fancyRpartPlot(treefit_pixel,tweak=1.5)
predicted_pixel_tree = predict(treefit_pixel,pixel_test_10,type = "class")

Results_pixel <- data.frame(predicted=predicted_pixel_tree,Actual=pixel_test_10_label)
(table(Results_pixel))
confusionMatrix(Results_pixel$predicted,Results_pixel$Actual)

printcp(treefit_pixel)
plotcp(treefit_pixel)


ptree<- prune(treefit_disc,
              cp= treefit_disc$cptable[which.min(treefit_disc$cptable[,"xerror"]),"CP"])

pruned_treefit <-fancyRpartPlot(ptree, uniform=TRUE,
                                main="Pruned Classification Tree")

####Ok not great, lets try descritizing the data################################################
################################################################################################
pixel_train <- read.csv(file = "Kaggle-train1400.csv",header = T,sep = ",",stringsAsFactors = F)
pixel_train3 <- pixel_train %>%
  mutate_all(as.numeric)
pixel_train3$label <-as.factor(pixel_train$label)
str(pixel_train3)

#ok this didn't work, but it showed this interesting error:
#The calculated breaks are: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 205, 253, 255
#we will need to use a fixed range
discritized_cuts <- discretize(pixel_train3[,2:ncol(pixel_train3)],method = "frequency",breaks = 20)
#but fist lets find number frequencies
max(pixel_train3[,2:ncol(pixel_train3)]) 
#so 255 is the max, which meaks sense, now lets 
c(1,80,205,253,255)
length(which(pixel_train3[,2:ncol(pixel_train3)]>=1 & pixel_train3[,2:ncol(pixel_train3)]<= 80))
length(which(pixel_train3[,2:ncol(pixel_train3)]>=81 & pixel_train3[,2:ncol(pixel_train3)]<= 205))
col_num <- c(1:255)
(col_freq <- length(which(2==pixel_train3[,2:ncol(pixel_train3)])))

#nothing working, lets just do a for loop
col_freq <- vector(length = length(col_num))
for (i in col_num) {
  col_freq[i]<-length(which(i==pixel_train3[,2:ncol(pixel_train3)]))
  i <- i+1

}

freq_dist<- data.frame(col_num,col_freq)
freq_dist
plot(freq_dist,type='h',xlab="Numbers 1-255", ylab="Frequency", main="Frequency Distribution of Pixel Numbers")
#ok interesting, lots of number frequencies that are near 255, which signifies the edge of numbers
#discretized_cuts <- discretize(pixel_train3[,2:ncol(pixel_train3)],
 #                              method = "fixed",breaks = c(-Inf,1,80,205,253,Inf))
pixel_disc_train <- discretizeDF(pixel_train3[,2:ncol(pixel_train3)],
                                 default = list(method='fixed',breaks=c(-Inf,1,80,205,253,Inf)))
pixel_disc_train <- cbind(pixel_train3$label,pixel_disc_train)
colnames(pixel_disc_train)[1] <- "label" 
pixel_disc_train$label

### Same Sample process as before
pixel_disc_train <- pixel_disc_train[-c(all_sample_vec),]
pixel_disc_test <- pixel_disc_train[c(all_sample_vec),]

NB_pixel_disc <- naiveBayes(label~.,data = pixel_disc_train,na.action = na.pass)
#NB_fit1 <- naive_bayes(pixel_train_10$pixel_label~.,data = pixel_train_10)

#prepare test data (remove labels)
pixel_disc_test_label <- pixel_disc_test$label


#first take the first 10 rows
pixel_disc_test <- pixel_disc_test[,-1]



#Now predict using prediction fromula
NB_pixel_disc_prediction <- predict(NB_pixel_train10,pixel_test_10)
#NBpred1 <- predict(NB_pixel_train10,as.data.frame(pixel_test_10))
#NB_pixel_train10

confusionMatrix(NB_pixel_disc_prediction,pixel_disc_test_label)
data.frame(NB_pixel_train10_prediction,pixel_test_10_label)


treefit_disc <- rpart(pixel_disc_train$label ~ ., data = pixel_disc_train, method="class")
summary(treefit)
#can't really see trying a new packaage
fancyRpartPlot(treefit_disc,tweak=1.5,nn.cex=.01)
?prp()

predicted_disc = predict(treefit_disc,pixel_disc_test,type = "class")
Results <- data.frame(predicted=predicted_disc,Actual=pixel_disc_test_label)
(table(Results))
confusionMatrix(Results$predicted,Results$Actual)

printcp(treefit_disc)
plotcp(treefit_disc)


ptree<- prune(treefit_disc,
              cp= treefit_disc$cptable[which.min(treefit_disc$cptable[,"xerror"]),"CP"])

pruned_treefit <-fancyRpartPlot(ptree, uniform=TRUE,
                main="Pruned Classification Tree")

predicted_disc = predict(pruned_treefit,pixel_disc_test,type = "class")

#This was a little better but lets mess with the boundaries:

pixel_disc_train_v2 <- discretizeDF(pixel_train3[,2:ncol(pixel_train3)],
                                 default = list(method='fixed',breaks=c(-Inf,1,25,50,75,100,125,
                                                                        150,175,200,225,250,Inf)))
pixel_disc_train_v2 <- cbind(pixel_train3$label,pixel_disc_train_v2)

colnames(pixel_disc_train_v2)[1] <- "label" 
pixel_disc_train$label

### Same Sample process as before
pixel_disc_train_v2 <- pixel_disc_train_v2[-c(all_sample_vec),]
pixel_disc_test_v2 <- pixel_disc_train_v2[c(all_sample_vec),]

NB_pixel_disc_v2 <- naiveBayes(label~.,data = pixel_disc_train_v2,na.action = na.pass)
#NB_fit1 <- naive_bayes(pixel_train_10$pixel_label~.,data = pixel_train_10)

#prepare test data (remove labels)
pixel_disc_test_label_v2 <- pixel_disc_test_v2$label


#first take the first 10 rows
pixel_disc_test_v2 <- pixel_disc_test_v2[,-1]



#Now predict using prediction fromula
NB_pixel_disc_prediction_v2 <- predict(NB_pixel_disc_v2,pixel_disc_test_v2)
#NBpred1 <- predict(NB_pixel_train10,as.data.frame(pixel_test_10))
#NB_pixel_train10

confusionMatrix(NB_pixel_disc_prediction_v2,pixel_disc_test_label_v2)
data.frame(NB_pixel_train10_prediction,pixel_test_10_label)

plot(NB_pixel_disc_prediction_v2)

#####3 The test
str(pixel_test)
pixel_test_dic <- discretizeDF(pixel_real_test, default = list(method='fixed',breaks=c(-Inf,1,80,205,253,Inf)))

pixel_test_dic
NB_prediction_pixel <- predict(NB_pixel_disc_v2,pixel_test_dic)


### NaiveBayes practice

## Example with metric predictors:
data(iris)
m <- naiveBayes(Species ~ ., data = iris)
## alternatively:
m <- naiveBayes(iris[,-5], iris[,5])
m
table(predict(m, iris), iris[,5])
# }


library(tidyverse)
#install.packages("janitor")
library(janitor)
CleanTrainset2<-janitor::clean_names(Trainset2)
# (CleanTrainset2[,c(upon, adil)])
# (table(CleanTrainset2$row_names))
# (table(Trainset2$RowNames))
# 
(NumCols<-ncol(Trainset2))   ## How many columns
(Num_sum<-sum(sapply(Trainset2, is.numeric))) 
(Num_sum<-sum(sapply(Trainset2, is.logical))) 
NBFit6<- naivebayes::naive_bayes(Trainset2$RowNames~., data=Trainset2)
(summary(NBFit6))
NBPred6 <- predict(NBFit6, Testset_noLab2)
### CONFUSION MATRIX
(table(TestLabels2,NBPred6))
### These are very poor results



## Example of using a contingency table:
data(Titanic)
m <- naiveBayes(Survived ~ ., data = Titanic)
m
predict(m, as.data.frame(Titanic))

testIndexes <- sample(1:nrow(pixel_train2), size=1*nrow(pixel_train2))
testIndexes

data.frame()
#####descritize practice
#this will allow you to figure out the the cuts for the dataframe, i.e frequencies
#then apply back to the whole data frame.
x1 <- data.frame(as.numeric(c(1:100)),as.numeric(c(50:149)))
x2<- discretize(x1,method = "frequency",breaks = 3,onlycuts = T)
x3<- discretizeDF(x1,default = list(method='fixed',breaks=x2))
length(which(x1>=1 & x1 <= 50))
naiveBayes(as.numeric.c.1.100.. ~ ., data = x3)
str(x3)

x1$bins <- as.numeric(cut(x1,breaks = x2)) 
x3 <- discretize(x1,method = 'fixed',breaks = c(x2))

discretize(x1, method = "frequency",breaks = 4,df=T)
str(x2)
str(x1)


