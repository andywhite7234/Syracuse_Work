install.packages("devtools")
library(devtools)          
library(RJSONIO)
library(rjson)  
library(jsonlite)
library(ggplot2)
#I found the most help from https://ropensci.org/tutorials/rnoaa_tutorial/
#Go here for more info on dataid information: https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
#Go here for station list: https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt
#in order to have access you need a token, you can "order" one from noaa, mine is: aRDlwGcDCHNulBwzeTmxCRBqhirzTKWK       
#help obtaining a key: https://ropensci.org/blog/2014/03/13/rnoaa/
#then run this code - this tells NOAA you have a token and can access database

library(rnoaa)
options(noaakey = "aRDlwGcDCHNulBwzeTmxCRBqhirzTKWK") 

#Step1: Get data into ISO date to be ingested by noaa. I chose 1900 as start point
data.df.index.start <- c(1900:2018)
data.df.index.start<- data.frame(data.df.index.start,"01","01")
colnames(data.df.index.start)<-c("Year","Month","Day")
data.df.index.start$startdate <- as.character(paste(data.df.index.start$Year,data.df.index.start$Month,data.df.index.start$Day,sep="-"))
data.df.index.start$enddate<- as.character(paste(data.df.index.start$Year,12,31,sep="-"))

#step2: Create function to return data for all years needed
#first rename data df. FUNCTION WILL NOT WORK IF THIS ISN'T DONE:
startend <- data.df.index.start
startend

#the following function loops through the RNOAA API and pulls based on specification, and datatype
#this allows for streamlined data pulling from their api
noaapull <- function(year1,datatype,stationid1){
  
  year1 <- match(year1,startend$Year)
  pullstart <- ncdc(datasetid = "GHCND", stationid=stationid1, datatypeid=datatype, startdate = startend[year1,4], enddate = startend[year1,5], limit=500)
  
  for (i in (year1+1):nrow(startend)) {
    if (i == (year1+1)){
      year2<- ncdc(datasetid = "GHCND", stationid=stationid1, datatypeid=datatype, startdate = startend[i,4], enddate = startend[i,5], limit=500)
      pullstart <- ncdc_combine(pullstart,year2)
    }else{
      yearx <- ncdc(datasetid = "GHCND", stationid=stationid1, datatypeid=datatype, startdate = startend[i,4], enddate = startend[i,5], limit=500)
      pullstart <- ncdc_combine(pullstart,yearx)
    }
  }
  pullstart<- pullstart
  return(pullstart[[1]])
}

#Finally make sure that you assign an object variable to the function:
test <- noaapull(2010,'AWND','GHCND:USW00023160')
ncdc_plot(test)
write.csv(test)
#Get data from a list to a DF with
metadata <- test[[1]]
write.csv(metadata, file=)
plot(metadata$value)
metadata <- data.frame(metadata)
metadata$value2 <- as.numeric(metadata$value)
str(metadata)
na.omit(metadata)
plot(metadata,x=metadata$date,y=metadata$value2)
date(metadata[1,1])
metadata <- as.data.frame(metadata)

metadata[1,1]
ggplot(metadata,aes(x=date,y=value2,group=value2))+geom_line()

#######OTHER HELPFUL NOTES:#######3

#Tucson Stations: USC00028815 (didn't work with percip) and USW00023160
#GHCND - stands for Global Historical Climate Network -Daily. Can also connect to GHCNM (monthly)
#CORE Elements: PRCP (tenths of mm), SNOW, SNWD (Snow depth), TMAX (tenths of degrees C), TMIN, TSUN (Daily total Sunshine Minutes), 
#PSUN (Perscent sun), AWND (Av Daily Wind Speed), AWDR (average daily Wind direction), 
#ACSH/ACSC (average cloudiness sunrise to sunset the ACSC is for 30-second ceilometer data)
#ACMC/ACMH (average cloudiness midnight to midnight)


#####THE FOLLOWING IS BUILDING/LEARNING TO GET TO ABOVE FUNCTION#######
#this will get info on a station by specifiying a datsetid, locationid and stationid
output <- ncdc_stations(datasetid = 'GHCND','locationid=FIPS:12017',stationid = 'GHCND:USC00084289')
output
#search for data nd get a data.frame. As reminder everything searched needs to be in quotes ""
out <- ncdc(datasetid='NORMAL_DLY', datatypeid='dly-tmax-normal', startdate = '2010-05-01', enddate = '2010-05-10')
#next get the data:
out$data

out <- ncdc(datasetid='NORMAL_DLY', stationid='GHCND:USW00023160', datatypeid='dly-tmax-normal', startdate = '2010-01-01', enddate = '2010-12-10', limit = 300)
out$data
ncdc_plot(out)
#ncdc plots data, hopefully we can use the ggplot down the road

#Now lets try precipitation:
out <- ncdc(datasetid='GHCND', stationid='GHCND:USW00014895', datatypeid='PRCP', startdate = '2010-05-01', enddate = '2010-10-31', limit=500)
ncdc_plot(out)

TucsonOut <- ncdc(datasetid='GHCND', stationid='GHCND:USW00023160', datatypeid='PRCP', startdate = '2010-05-01', enddate = '2010-10-31', limit=500)
ncdc_plot(TucsonOut)
TucsonOut$data

#Tucson Stations: USC00028815 (didn't work with percip) and USW00023160
#GHCND - stands for Global Historical Climate Network -Daily. Can also connect to GHCNM (monthly)
#CORE Elements: PRCP (tenths of mm), SNOW, SNWD (Snow depth), TMAX (tenths of degrees C), TMIN, TSUN (Daily total Sunshine Minutes), 
#PSUN (Perscent sun), AWND (Av Daily Wind Speed), AWDR (average daily Wind diretion), 
#ACSH/ACSC (average cloudiness sunrise to sunset the ACSC is for 30-second ceilometer data)
#ACMC/ACMH (average cloudiness midnight to midnight)
#Go here for more: https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
#Go here for station list: https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt

statURL <- "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
apiResult <- getURL(statURL)

#First test functions for NOAA
prcepitation <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='PRCP', startdate = '2010-05-01', enddate = '2010-10-31', limit=500)         
tempmax <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TMAX', startdate = '2018-05-01', enddate = '2018-10-31', limit=500)
tavg <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TAVG', startdate = '2018-05-01', enddate = '2018-10-31', limit=500)
ncdc_plot(tavg)
tavg <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TAVG', startdate = data.df.index.start[118,4], enddate = '2017-12-31', limit=500)          
psun <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TMAX', startdate = '2000-01-01', enddate = '2000-11-31')
ncdc_plot(tavg)

#the plots work, now lets test the ncdc function, to see if it can take a variable:
strtdate <- '2018-05-01'
str(data.df.index.start[119,4])

#does work, now lets create a loop. Idea data frame with each year and then loop through each vector and run code:

#Part 1: Get data into ISO date to be ingested by noaa
data.df.index.start <- c(1900:2018)
data.df.index.start<- data.frame(data.df.index.start,"01","01")
colnames(data.df.index.start)<-c("Year","Month","Day")
data.df.index.start$startdate <- as.character(paste(data.df.index.start$Year,data.df.index.start$Month,data.df.index.start$Day,sep="-"))
data.df.index.start$enddate<- as.character(paste(data.df.index.start$Year,12,31,sep="-"))
strte

#Step2: Next test the combine function to have longer dataframes
tavg <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TAVG', startdate = data.df.index.start[118,4], enddate = data.df.index.start[118,5], limit=500)          
tavg2 <- ncdc(datasetid = "GHCND", stationid='GHCND:USW00023160', datatypeid='TAVG', startdate = data.df.index.start[119,4], enddate = '2018-12-31', limit=500)          
df.tavg <- ncdc_combine(tavg,tavg2)
#test checks out! Two years of data combinedncdc_plot(df.tavg)

#step3: Create function to return data for all years needed
#first rename data df:
startend <- data.df.index.start
startend

# Tucson station ID: GHCND:USW00023160

#Step4: create function:
noaapull1 <- function(year1,datatype,stationid1){
  
  year1 <- match(year1,startend$Year)
  pullstart <- ncdc(datasetid = "GHCND", stationid=stationid1, datatypeid=datatype, startdate = startend[year1,4], enddate = startend[year1,5], limit=500)
  
  for (i in (year1+1):nrow(startend)) {
    
    yearx     <- ncdc(datasetid = "GHCND", stationid=stationid1, datatypeid=datatype, startdate = startend[i,4], enddate = startend[i,5], limit=500)
    
    pullstart <- ncdc_combine(pullstart,yearx)
    
  }
  pullstart<- pullstart[[1]]
  write.csv(pullstart,file = paste(datatype,year1,stationid1))
  
  return(read.csv(paste(datatype,year1,stationid1)))
}

test <- noaapull(2010,'AWND','GHCND:USW00023160')
test1 <- noaapull1(2010,'AWND','GHCND:USW00023160')
str(test1)
plot(test1, y=test1$value,x=test1$date)
ncdc_plot(test)
attemp <- "fun"
attemp2<-"sorta"

#Get data from a list to a DF with
metadata <- test[[1]]
metadata
metadata <- data.frame(metadata)
write.csv(metadata,file=paste(attemp,attemp2))
read.csv(paste(attemp,attemp2))
plot(metadata$value)

stationlist <- ncdc_stations()
str(stationlist)
stationlist[[1]]
