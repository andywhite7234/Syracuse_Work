
##############################Final Class###########################################
########################### You Made it!!! ######################################
##############################################################################

#We will switch gears a little from the viz and do some predictions. Most of this will be data cleaning. 
# this part never goes away. How do we clean and subset data. 
#install.packages('rpart')
#install.packages('rattle')
#install.packages('rpart.plot')
#install.packages('arules')
#install.packages('GGally')
#install.packages('caret')
library(GGally)   #for ggpairs plot
library(rpart)
library(rattle)
library(rpart.plot)
library(ggplot2)
library(arules)
#install.packages("readxl")
library(readxl)
library(caret)
######## PLEASE UPDATE WITH YOUR FILE LOCATION ######################################

default <- read_excel("C:/Users/andy_white/Desktop/Projects/R/SP_course/SP_Training/CMBS_Practice_Data.xlsx",
                      sheet = "default", col_names = T)
#View the dataframe:
View(default)
str(default)
#call the following columns. How did i know to call the columns? I took a look at the data in excel and in the view
#
default$`Loan status at maturity`
default2 <- data.frame(Loan_stat = default$`Loan status at maturity`,Vintage=default$VINTAGE, Prop_type=default$PROPERTY_TYPE,                        Appraised=as.numeric(default$APPRD_VAL_AMT),Loss_sev=as.numeric(default$LOSS_AMT),
                       DY=as.numeric(default$`Debt Yield for Most recent available period`),Orig_bal=as.numeric(default$LOAN_ORIG_BAL_AMT),
                       City=default$CITY,State=default$ST_CD,SQFT = default$NET_SQFT)


#ALL IS A COPY FROM LAST TWO CLASSES
default2 <- default2[!(default2$DY=="NO DY"),]
#Calculate a loan to value: simply take the loan appount vector and divide by appraisal amount vector
default2$LTV <- default2$Orig_bal/default2$Appraised

#Things tend to go a little haywire in the viz if you have N/A's there are multiple strategies,
# but for this purpose just remove. The ! does this. Also sometimes the - will also, but generally is
# for removing string types. N/A is actually not a string in R. 

summary(default2)
default2 <- default2[!is.na(default2$DY),]
default2 <- default2[!is.na(default2$Appraised),]
default2 <- default2[!is.na(default2$Loss_sev),]
default2 <- default2[!is.na(default2$SQFT),]
rownames(default2) <- NULL


ggpairs(default2[,c(1,3,4)])
####so these are ugly numbers, lets refresh one time:
options(scipen = 1000000)

#now lets retry
ggpairs(default2[,c(1,3,4)])

#lets create a correlation matrix:
res <- cor(default2[,c(4,5,6,7,10,11)])
res
round(res,2)

## now lets do some regression:
fit1 <- lm(Loss_sev ~ LTV + DY + SQFT + Orig_bal + Appraised + State + Prop_type + 
             Vintage,data = default2)
summary(fit1)
?lm
##What's the difference between correlation and regression? Correlation shows if two variables move together
# from our matrix we see items that move together are loss severity and loan balance - not really surprising
# as loans get larger we would expect the loss severity to be greater. We also see that as Debt yeild decrease
# loss severity increases. However it doesn't quite help us to make a prediction. That's where regression comes in handy

#Regression give you find significant variables that impact Loss severity and ultimately spit out
# a "prediction" equation. 

#### Now lets do some far more advanced prediction, which you'll find is very easy to run through
### once you clean your data


##### Decision Tree #########

#a couple notes on 
## Next, we will use Decision Trees to see if we can classify data by loss severity

## Decision Trees will run best if I discretize the data. What is discretizing?
## - It's taking numeric data and changing it to categorical data. 
## - However it is not using an equation like as.factor() or as.charachter()
## - As a simple example: if we have 100 loans with appraisal values of 1,2,3... all the way to 100
##      - We would "cut" the data set into categories like 1-10, 10-20, etc
## I will also need to use Testing and Trainng datasets. But First Discretize:
summary(default2)
str(default2)

default3 <- default2[,c(-3,-8,-9)] #assign default 2 to a new variable and remove prop type, city, and state
#tree will be too complex with 50 different states and 100 different cities

#we need to normalize loss severity and the best way to do that is in % of loan balance
default3$Loss_sev <- default3$Loss_sev/default3$Orig_bal

#now lets descretize the numerical data:
default3$Loss_sev <- discretize(default3$Loss_sev, breaks = 10)
head(default3)
summary(default3$Loss_sev)

# lets use same equation for all numerical columns:
# big thing - once you discretize you cannot change back to numerical - I alwasy recommend creating a new variable
str(default3)
default3$Appraised <- discretize(default3$Appraised,breaks = 10)
default3$Orig_bal <- discretize(default3$DY,breaks = 10)
default3$LTV <- discretize(default3$LTV,breaks = 10)
default3$SQFT <- discretize(default3$SQFT,breaks = 10)
default3$DY <- discretize(default3$DY,breaks=10)


### Ok now we have all categorical data and are almost ready to feed the decision tree algo
# generally I like to have the training set somewhere around 60-90% of the data
# and the test set around 40-10% of the data here's a neat function that creates a sample for every #nth row

(DF_length <- 1:nrow(default3))
sample_length <- DF_length[seq(2,length(DF_length),10)]
default3[c(2,12,22),]
#### Now split default3 that has been discretized into a training and testing set:

default3_test <- default3[c(sample_length),]
default3_test
default3_train <- default3[-c(sample_length),]
default3_train

#ok now we are ready for decision tree algo
#first you need to train the algorithm and let it know the predictor is going to be loan_stat
default_tree <- rpart(default3_train$Loan_stat ~.,data=default3_train,method = 'class' )
summary(default_tree)
rpart.plot(default_tree)
#a little hard to see
fancyRpartPlot(default_tree,type=1,cex=.4)
#loss severity seems to be a good predictor. which as a lagging indicator. Meaning a loan defaults
#and then eventually a loss severity comes through. Lets prune the data set and remove loss severity from
# the dataframe
default3 <- default3[,-4]
#re split the dataset
default3_test <- default3[c(sample_length),]
default3_test
default3_train <- default3[-c(sample_length),]
default3_train

default_tree <- rpart(default3_train$Loan_stat ~.,data=default3_train,method = 'class' )
summary(default_tree)
rpart.plot(default_tree)

#now lets see how the predictor does:
loss_test_labels <- default3_test$Loan_stat
default3_test <- default3_test[,-1]  #remove the class of loan status (term default, maturity default)


treefit_pred <- predict(default_tree,default3_test,type="class")
(results <- data.frame(Predicted = treefit_pred,Actual = loss_test_labels))

table(treefit_pred,loss_test_labels)



iris_tree <- rpart(iris$Species ~.,data=iris, method = 'class')
fancyRpartPlot(iris_tree)
summary(iris_tree)

DF_length <- c(1:nrow(iris))

sample_length <- DF_length[seq(2,length(DF_length),7)]

#### Now split default3 that has been discretized into a training and testing set:
sample_length
iris_test <- iris[c(sample_length),]

iris_train <- iris[-c(sample_length),]

iris_test_label <- iris_test[,5]
iris_test_label <- as.character(iris_test_label)
iris_test<- iris_test[,-5]

iris_tree <- rpart(iris_train$Species ~.,data=iris_train, method = 'class')
pred_iris <- predict(iris_tree,iris_test)
confusionMatrix(pred_iris,iris_test_label)

?predict









