####R has a number of internal datasets - these are important to test whatever
### machine learning technique you will be involved in. Iris is great
#first you have to call the dataset:
data("iris")
2+2

#This function reads in thegoogle correlate data, with the headers
read.csv('google_correlate.csv', header = T)
?read.csv2

#This does the same thing, a little longer write-out, but also useful
#note I'm assigning the filepat to the variable file_path_google
file_path_google <- 'C:/Users/andy_white/Desktop/Projects/R/SP_course/SP_Training/google_correlate.csv'
read.csv(file_path_google,header = F)

#Try reading in the headers as false and see what happens

#Call the dataset Iris. You cannot proceed unless you have data("iris")
data("iris")
iris   # Ctrl+Enter    to view the data set

###the str is a fantastic function ti get to know. This will display the structure of
###a variable of your choosing
str(iris)

###Also if you are unfamiliar with a function a great way to get a little info
## put a ?str. Execute the code below to see
?str

# now with Iris dataframe call the column (vector) Petal.Width
iris$Petal.Width

#function to convert to factor
as.factor(iris$Petal.Width)
str(iris)
#notice here that the actual dataframe was not converted to a factor. 
#You need to specify the dataframe and column and then assign to a factor:
iris$Petal.Width <- as.factor(iris$Petal.Width)

#This reassigns the Petal.Width column within the Iris dataframe to a factor, 
#Try your str function to confirm:
str(iris)

#what is a factor? Factors in R are stored as a vector of integer values 
#with a corresponding set of character values to use when the factor is 
#displayed. Sort of a 3 dimensional column of data that has two values corresponding to it


#Now lets try to convert back to numeric
iris$Petal.Width <- as.numeric(iris$Petal.Width)
str(iris)

#now notice the slight difference between numeric and integer
iris$Petal.Width <- as.integer(iris$Petal.Width)
str(iris)
#if you look in the petal.width - integar only shows 2 vs numeric would show 2.1
#But BE CAREFUL with converting to and from integer. notice that when we try
#to convert back to numeric, the decimal points are dropped
iris$Petal.Width <- as.numeric(iris$Petal.Width)
str(iris)
#now recall iris:
data("iris")
#boom data is back to normal

#now sum up sepal lenght - this will sum up the vector
sum(iris$Sepal.Length)
sum(iris[,1:3])
#notice that you can sum multiple vectors (columns) of data with very little 
#modification

##Another useful function. take a look at the first 5 rows
head(iris)

#This will take a look at the first 15 rows
head(iris,15)

#find the row length or count of dataframe (rowsum)
length(iris$Sepal.Length)

# another handy way - find all values for which
length(which(iris$Sepal.Length==5.1))
#which returns the row of the logical specification, below we find rows equal to 5.1
which(iris$Sepal.Length==5.1)

#you can also create a new dataframe with only iris sepal.lengths of 5.1
iris2 <- iris[which(iris$Sepal.Length==5.1),]
iris2
#also try the view function - it's great at interacting with the dataset
View(iris2)
#breaking up a dataframe/Accessing a fow or column
#take a column
iris[,1]
#take a row:
iris[1,]
#grab rows 50 - 60
iris[50:60,]
View(iris)
#Take cell from row 5 and column 5
iris[5,5]
summary(iris)
#assign the follwing to hello worlds 1,2,3
hello <- "hello world"
hello2 <- "hello world"
hello3 <- "HELLO world"

#test out the logical
hello == hello2
hello == hello3

#A handy function to lowercase all letters in a specified string
#this function is really handy in text mining
tolower(hello3)
hello == hello3

hello3 <- tolower(hello3)
hello == hello3

#Try out these logical
1 >2 & 3 > 4
1 <2 & 3 >4
1 <2 & 3 <4

#or
1>2 |  3 >4
1<2 | 3>4
1 < 2 | 3 < 4
#now you can use everything combined to find sepal length between 5 and 7
which(iris$Sepal.Length>5 & iris$Sepal.Length <7)


#some other useful packages
install.packages("censusapi")
library(censusapi)
library(sqldf)
library(RSQLite)

#code from HUB post:
#don't forget the comma near the end of the brackets - this one hurts when you're missing the comma, and it takes you hours to figure out
iris3 <- iris[which(iris$Sepal.Length >6),]
#now check:
View(iris3)
#you can also check using the sum function (works on logical vectors)
sum(iris$Sepal.Length >6)
#as a check
sum(iris$Sepal.Length >7)


#length(iris3$Sepal.Length)


