####Class 3 ########
#### We will be continuing viz and data manipulation
#### We will conncect to the census api and then visualize the data:

install.packages("ggmap")
install.packages("censusapi")
install.packages("tidycensus")
install.packages("tigris")
#install.packages("devtools")   #for zip codes
install.packages("remotes")
install.packages("viridis")
install.packages("dplyr")
install.packages("ecb") #for the European Maps and stats
install.packages("leaflet")
install.packages("tidycensus")
library(ggplot2)
library(ggmap)
library(censusapi)
library(tidycensus)
library(tigris)
library(remotes)
library(ecb)
library(viridis)   #for coloring
library(dplyr)
library(zipcode) #depreciated version, please see below if zipcode download is giving you an error
library(leaflet)
??install_version
install_version()
install_version("zipcode",version = "1.0")
library(zipcode)
library(tidycensus)
options(scipen = 1000000)
data(zipcode)
zipcode1 = clean.zipcodes(zipcode$zip)
str(zipcode1)
str(zipcode)
write.csv(zipcode,file='zipcodes.csv')
#zipcode::clean.zipcodes()
#below is my census_api_key. IT WILL NOT WORK ON YOUR MACHINE. You need to go here:
# https://api.census.gov/data/key_signup.html
census_api_key("a70a834d95b1f51c92fa37997a7a70da54dbdbc7",install = T)
census_api_key("a70a834d95b1f51c92fa37997a7a70da54dbdbc7",install = T)
#make sure to have install=True like above, it simply means your key has been added to tinycensus package
#I also have it saved to a variable like below:
mycensuskey <-"a70a834d95b1f51c92fa37997a7a70da54dbdbc7"

#if you didn't
v17 <- load_variables(2017, "acs5", cache = TRUE)
View(v17)
write.csv(v17,file = 'C:/Users/andy_white/Desktop/Projects/Syracuse/IST 718/Labs/Lab 2/census_data_info.csv')

#if you want to grab data for a specific state and county
#fantastic info related to tidycensus:
# https://walkerke.github.io/tidycensus/articles/basic-usage.html

#state population grab for just Mizz:
state_pop <- get_acs(geography = "state", 
                   variables = "B01003_001", state = "MO", #county = "Wright",  
                   geometry = TRUE, year = 2017) #county = "Wright")
state_pop

#we can also grab state level income. 
us_state_income <- get_acs(geography = "state", variables ="B05010_001",
                           geometry = T)    #this allows for shift of HI and AK onto map
trying = get_acs(geography = "zcta", variables ="B05010_001",
                 geometry = F)
#this is a new type of data we've never seen before: we now have a list:
head(us_state_income)
str(us_state_income)

#lets try a quick ggplot viz:
ggplot(data=us_state_income)+
  geom_sf(aes(fill=estimate))


#What's happening? its trying to plot the coordinates of guam, puerto rico, AK and HI. So the viz is
#is all messed up. Two ways to handle. The easy way (filtering out and very important to know) and 
#the real easy way for this particular problem
View(us_state_income)

#leave out AK, HI, and PR (state FIPS: 02, 15, and 72)
default2 <- default2[!is.na(default2$DY),]
View(us_state_income)
us_state_income <- us_state_income[!(us_state_income$GEOID %in% c("02","15","72")),]  
#ok now lets viz one more time:
ggplot(us_state_income)+
  geom_sf(aes(fill=estimate))

#BOOM! ok now lets go the real easy way. 
us_state_income <- get_acs(geography = "state", variables ="B19013_001",
                           shift_geo = T,geometry = T)    #this allows for shift of HI and AK onto map

ggplot(us_state_income)+
  geom_sf(aes(fill=estimate))

#lets get county level data:
us_county_income <- get_acs(geography = "county", variables = "B19013_001", 
                            shift_geo = TRUE, geometry = TRUE)
us_county_income
ggplot(us_county_income)+
  geom_sf(aes(fill=estimate))

#shift geo is only available for states and county level. so need to remove
us_zip_income <- get_acs(geography = "zcta", variables = "B19013_001", 
                          geometry = TRUE)
us_zip_income

#way too hard to vizualize on a US map - but lets drill down to the state level:

#we need to merge zipcode (so we have state names and we even get county names if we want to drill
#down to the that level)
#   merge(x,y,by.x="geoid",by.y="zip")

us_zip_income2 <- merge(us_zip_income,zipcode,by.x=c("GEOID"),by.y=c("zip"))
#a quick check:
head(us_zip_income2)
#ok lets filter by NY:
NY_zip_income <- us_zip_income2[(us_zip_income2$state %in% c("NY")),]
NY_zip_income
#one more viz:
ggplot(NY_zip_income)+
  geom_sf(aes(fill = estimate))

###finally lets demo leaflet. Interactive map shaded by zip code
pal <- colorNumeric("viridis", NULL)
leaflet(NY_zip_income) %>%
  addTiles() %>%
  addPolygons(stroke = FALSE, smoothFactor = .3,fillOpacity = 1,
              fillColor = ~pal(log10(estimate)),
              label = ~paste0(GEOID, ": ", estimate)) %>%
  addLegend(pal = pal, values = ~log10(estimate), opacity = 1.0,
            labFormat = labelFormat(transform = function(x) round(10^x)))

##########################################################################################
#### Code from last class

#remember our default data? Lets plot it on here too:
default2 <- data.frame(Loan_stat = default$`Loan status at maturity`,Vintage=default$VINTAGE, Prop_type=default$PROPERTY_TYPE,                        Appraised=as.numeric(default$APPRD_VAL_AMT),Loss_sev=as.numeric(default$LOSS_AMT),
                       DY=as.numeric(default$`Debt Yield for Most recent available period`),Orig_bal=as.numeric(default$LOAN_ORIG_BAL_AMT),
                       City=default$CITY,State=default$ST_CD,Zip=default$ZIP_CD)
colnames(default2)
default2[1:10,1:8]
summary(default2)
str(default2)
#Ok now a little bit of that cleaning
#this will remove all values in the DY column that have "No DY"
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
default2 <- default2[!is.na(default2$Zip),]
default2 <- default2[-which(default2$Zip=="XX"),]

summary(default2)
rownames(default2) <- NULL
#
#need to merge zipcode file into default (need lat/lon)
default_zip<- merge(default2,zipcode, by.x = "Zip",by.y = "zip")

#check to see if there are any NAs:
summary(default_zip)

#filter (or as r users say subset) NY's or else we will have the same problem when plotting Puerto rico, ak and HI
#we now have two state columns which I am too lazy to clean 
NY_default <- default_zip[(default_zip$State %in% c("NY")),]


#now lets plot vs our dope map:
ggplot(NY_zip_income)+
  geom_sf(aes(fill = estimate))+
  geom_point(data=NY_default,aes(y=latitude,x=longitude),color="red")

state.abb
data("state")
state
ggplot()

?heatmap
#lets make sure that 
#Cool looking color scheme:
ggplot(us_county_income) + 
  geom_sf(aes(fill = estimate), color = NA) + 
  coord_sf(datum = NA) + 
  theme_minimal() + 
  scale_fill_viridis_c()


?scale_fill_viridis



testing <- tigris::county_subdivisions(state = "MO")

iris2 <- iris
iris2
iris2 <- data.matrix(iris2)
iris2
str(iris2)

heatmap(iris2,scale = "column")
t(iris)
data <- as.matrix(mtcars)
heatmap(data)

##
us <- map_data("state")
dummyDF <- data.frame(state.name,stringsAsFactors = F)
dummyDF$state <- tolower(dummyDF$state.name)
dummyDF$state.abb <- state.abb

state_inc_DF <- data.frame(state.name=us_state_income$NAME,Med_inc = us_state_income$estimate)

dummyDF2 <- merge(dummyDF,state_inc_DF)

map.simple <- ggplot(dummyDF2,aes(map_id=state,colour="black")) + 
  geom_map(map=us,aes(fill=dummyDF2$Med_inc))

map.simple <- map.simple+expand_limits(x = us$long, y=us$lat)+coord_map()
map.simple





##################################################################################################
##################################################################################################
######################################### ECB Stats#################################################
##################################################################################################
##################################################################################################
#### Still a work in process
install.packages("eurostat")
library(eurostat)

#try oecd

#add tmap and tmap tools 
install.packages("tmap")
install.packages("tmaptools")
install.packages("rgdal")
install.packages(c("cowplot", "googleway", "ggplot2", "ggrepel", 
                   "ggspatial", "libwgeom", "sf", "rnaturalearth", "rnaturalearthdata"))
install.packages("rnaturalearth")
library(tmap)
library(tmaptools)
library(rgdal)
library(leaflet)
library(rnaturalearth)


# the following code shows my thought process building to a european map with shading:
#execute the code to see how it works
data(World,metro)
europe <- World[World$continent %in% "Europe",]
#this should filter only countires in Europe from the world dataset
europe 
#looks good now lets quickly vis in tmap 
ttm() #set the map mode to interactive viewing and then:
tm_shape(europe)+    #call the dataset, this must be done first with tm_shape, regardless which type of viz you use with tmap
  tm_polygons("HPI",id="name") #indicate that you want polygons - because you are drawing a map with polygon shapes
View(europe)
#ok viz shows mostly Europe, but it looks like it gets French Guiana, and viewing dataset suggests
# that that is built into the world dataset - so that doesn't quite work
world <- ne_countries(scale = "small",returnclass = "sf")
#lets try again:
europe <- world[world$region_un %in% "Europe",]
View(europe)
tm_shape(europe)+
  tm_polygons("pop_est",id="name")
  
#didn't quite work, lets try updating the projection with lat long coordinates,
#and then 
tm_shape(europe,projection = "longlat",xlim=c(-19,40),ylim = c(35,72))+
  tm_polygons("pop_est",id="name")

#Nice the latlon worked, but the picture quality is low. Lets change the data source back
data(World,metro)
europe <- World[World$continent %in% "Europe",]

tm_shape(europe,projection = "longlat",xlim=c(-23,40),ylim = c(35,72))+
  tm_polygons("HPI",id="name")
#looks good, now add in turkey to the dataset
turkey <- World[World$name %in% "Turkey",]
#two different ways to project on the map (layering appraoch) or modify dataframe
#layer:
tm_shape(europe,projection = "longlat",xlim=c(-23,40),ylim = c(35,72))+
  tm_polygons("HPI",id="name")+
tm_shape(turkey)+ ##add the turkey polygon shape on top of europe
  tm_polygons("HPI",id="name",legend.show =FALSE)

#layering is simple, but drawback is you need data to be same, and messing with the legend is tough
# and 
#### Second appraoch:
#modify DF:
europe <- rbind(europe, turkey) # rbind stands for row bind
tm_shape(europe,projection = "longlat",xlim=c(-23,42),ylim = c(35,72))+
  tm_polygons("HPI",id="name")
#great now we have the base layer
#lets layer some data from eurostat


# Get Eurostat data listing
toc <- get_eurostat_toc()
View(toc)
#I picked the sixth result
search_eurostat("Consumers - monthly data",
                type = 'table')$code[1]
search_eurostat("Modal split of passenger transport", 
                type = "table")$code[1]
world <- ne_countries(scale = "medium", returnclass = "sf")
#also can plot with GGPLOT
ggplot(europe)+
  geom_sf()+
  coord_sf(xlim=c(-19,40),ylim = c(35,72))

#that calls the world and metro dataset


df <- get_dataflows()


# Check the first items
head(toc)

pass <- head(search_eurostat("passenger transport"))

#Downloading Data, first get the id number:
id <- search_eurostat("Volume of passenger transport relative to GDP", 
                      type = "table")$code[1]

#ok now search for unemployment rate, and find code as a check
unemp <- search_eurostat("Unemployment rate",
                         type = "table")$code[1]
#returns unemployment code - this is based on county or small district refrence

#either copy and paste from view or you can grab since it's the 7th row in second column

#(id <- unemp[7,2])
print(id)

#now that we have the id number, we will use it to query the Eurostat db
dat <- get_eurostat(unemp, time_format = "num")
str(dat)
head(dat)

#this allows us to get nice looking geo labels for our countries
dat1 <- label_eurostat(dat)
head(dat1)
#levels(dat1$vehicle)
#convert dat to dataframe as we don't need to convert long names due to merge
dat1 <-data.frame(dat)
#filter by whatever year you want, or you can calc an average
dat1 <- dat1[dat1$time==2012,]

#there's a way to get to country level, now we just need to merge datasets
sf <- get_eurostat_geospatial(output_class = "sf", resolution = "60", nuts_level = "0")
counties <- get_eurostat_geospatial(output_class = "sf", resolution = "60", nuts_level = "2")

qtm(counties) #quick viz

#merge the datasets based on the same data
merged_data <- merge(sf,dat1,by.x="geo",by.y="geo")
(merg2 <- merge(counties,dat,by.x="geo",by.y="geo"))
#this is similar to a vlookup in excel

#now to viz
tm_shape(merg2,projection = "longlat",xlim=c(-23,42),ylim = c(35,72))+
  tm_polygons('values')

#not bad lets add a base layer with missing countries (from our "europe" dataset)
tm_shape(europe,projection = "longlat",xlim=c(-23,42),ylim = c(35,72))+
  tm_polygons()+
tm_shape(merged_data)+
  tm_polygons('values')
#now lets take year 2018:



# Get monthly data on annualized euro area headline HICP
hicp <- get_data("ICP.M.U2.N.000000.4.ANR")
head(hicp)

hicp_dims <- get_dimensions("ICP.M.U2.N.000000.4.ANR")
hicp_dims[[1]]
head(hicp_dims)



