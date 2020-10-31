library(mapdata)
library(readxl)
library(dplyr)
library(shiny)
library(readxl)
library(zipcode)
library(leaflet)
library(mapproj)
library(tmap)
library(tmaptools)
library(sf)
library(ggplot2)
library(ggmap)
library(ggrepel)
library(plyr)
library(scales)
library(devtools)
library(censusapi)
library(tidycensus)
library(stringr)
library(readr)
library(sqldf)
library(data.table)
library(maptools)
library(rgeos)
library(sp)
library(rgdal)
library(grid)
library(cartogram)
library("rnaturalearth")
library(osmar)
library(OpenStreetMap)
library(shinyjs)
library(spData)
library(s2)
library(spDataLarge)
library(tibble)
library(broom)
library(httr)
library(tigris)

#install.packages("s2")
data("World")
data("land")
data("metro")
data("NLD_muni")
data("rivers")
options(scipen = 100000)
states_tm <- data.frame(US_state2$NAME)
#to match with data sets

install.packages("shinyjs")
install.packages("cartogram")
install.packages('rgeos', type='source')
install.packages('rgdal', type='source')
options(scipen = 100000)

#run this function - need for later
Numberize <- function(inputVector){
  inputVector<-gsub(",","",inputVector)
  inputVector <-gsub(" ","",inputVector)
  return(as.numeric(inputVector))}

#This downloads the county map
f <- tempfile()
download.file("http://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_050_00_20m.zip", destfile = f)
unzip(f, exdir = ".")
US <- read_shape("gz_2010_us_050_00_20m.shp")
US2<-readShapeSpatial("gz_2010_us_050_00_20m.shp")

#filter out by state for publishing
NJ_data <- filter(US, STATE %in% c("34","36"))
NJ_data2 <- NJ_data[!(NJ_data$NAME %in% c("Bronx County","Kings County","Nassau County","Richmond County")),]

#this is a download of the states files
s <- tempfile()
download.file("http://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_040_00_20m.zip", destfile = f)
unzip(f, exdir = ".")
US_state2 <- read_shape("gz_2010_us_040_00_20m.shp")
# leave out AK, HI, and PR (state FIPS: 02, 15, and 72)
US <- US[!(US$STATE %in% c("02","15","72")),]  
US_state2 <- US_state2[!(US_state2$STATE %in% c("02","15","72")),]

# for theme_map
devtools::source_gist("33baa3a79c5cfef0f6df")

# nice US map GeoJSON
us11 <- readOGR(dsn="http://eric.clst.org/wupl/Stuff/gz_2010_us_040_00_500k.json", layer="OGRGeoJSON")

# even smaller polygons
us11 <- SpatialPolygonsDataFrame(gSimplify(us11, tol=0.1, topologyPreserve=TRUE), 
                                 data=us@data)

# don't need these for the continental base map
us <- us[!us$NAME %in% c("Alaska", "Hawaii", "Puerto Rico", "District of Columbia"),]

# for ggplot
map <- fortify(us, region="NAME")

#read in data:
weworkxl<-"C:/Users/andy_white/Desktop/Projects/WeWork Paper/Costar Data/USA WeWork.xlsx"
#wewrk cmbsw exposure
wewrkcmbs<-"C:/Users/andy_white/Desktop/Projects/WeWork Paper/WeWork/WeWork_TREPP_Prop_Level_Extract_20190327.xlsx"

#in order to get tmap to work, need to plot to different markets:
mttMSA <- "C:/Users/andy_white/Desktop/Projects/WeWork Paper/FREMF 2019-K87_MTT_11_28_2018.xlsx"

#get sheet one with sf totals into dataframe
wewrk <-data.frame(read_xlsx(weworkxl,sheet="Sheet1"))
wewrkcmbs<-data.frame(read_xlsx(wewrkcmbs,sheet = "Sheet2"))

#keep columns that are needed
wewrk <- wewrk[,c(2,3,4,5,6)]
#omit NA's that don't have sq ft totals
wewrk <- na.omit(wewrk)
#lowercase (doesn't have the state name for Washington DC)
wewrk$statename<- tolower(state.name[match(wewrk$State,state.abb)])
#turn sqft columns into number format:
wewrk$SF.Occupied<-Numberize(wewrk$SF.Occupied)

##### preparing data to plot on tmap ######
msa<-data.frame(read_xlsx(mttMSA,sheet = "MSA & CBSA Lookup"))

#remove uneeded rows
msa <- msa[,c(-2,-3,-5,-6,-8)]

#remove na's from remaining data
msa <- na.omit(msa)
rownames(msa)<-NULL
#rename columns:
colnames(msa)<- c("zip","city.state","city")

#bring in zip data base to "clean zips"
data("zipcode")

#turn first column into number, and clean the data:
msa$zip <- clean.zipcodes(msa$zip)

msa$city <- replace(msa$city,msa$city=="Long Island","New York")
msa$city <- replace(msa$city,msa$city=="Newark","New York")
msa$city <- replace(msa$city,msa$city=="San Jose","San Francisco")
msa$city <- replace(msa$city,msa$city=="Oakland","San Francisco")
msa$city <- replace(msa$city,msa$city=="Dallas/Fort Worth","Dallas")
msa$city <- replace(msa$city,msa$city=="Orange County","Los Angeles")
msa$city <- replace(msa$city,msa$city=="Riverside","Los Angeles")

#This will merge your wewrk zips with cmbs zips
wewrk = merge(wewrk,msa,by.x="Zip",by.y="zip")
wewrkcmbs=merge(wewrkcmbs,msa,by.x="ZIP",by.y="zip")

#Remove Dup columns
wewrk <-wewrk[,c(-3,-6)]
wewrkcmbs <- wewrkcmbs[,-9]
#now we just want our city and sf occupied to run a tapply
wewrk2 <- wewrk[,c(4,6)]
wewrkcmbs2 <- wewrkcmbs[,c(2,9)]
#calc total sq ft per city

wewrk2<- data.frame(tapply(wewrk2$SF.Occupied, wewrk2$city, sum))
wewrkcmbs2 <-data.frame(tapply(wewrkcmbs2$we.work.sqft,wewrkcmbs2$city,sum))

#get city names back prepare data to load into geocode function
wewrk2$city <- rownames(wewrk2)
rownames(wewrk2)=NULL
colnames(wewrk2)=c("sqft_tot","city")
wewrk2 <- merge(wewrk2,wewrk,"city")
                              #wewrk2$City <- tolower(wewrk2$City)

wewrk2<-na.omit(wewrk2)
#get lat lon for each major city:
wewrk2$geoload<-paste(wewrk2$city,wewrk2$State,sep=", ")
geoCode<- geocode(wewrk2$geoload,source="dsk")

#now cMBS:
wewrkcmbs2$city<-rownames(wewrkcmbs2)
rownames(wewrkcmbs2)<-NULL
colnames(wewrkcmbs2)=c("sqft_tot","city")
wewrkcmbs2<-merge(wewrkcmbs2,wewrkcmbs,"city")

geocmbs <- geocode(paste(wewrkcmbs2$city,wewrkcmbs2$STATE,sep = ", "),source = "dsk")
#put back into dataframe
wewrk2 <-data.frame(c(wewrk2,geoCode))
wewrkcmbs2<-data.frame(c(wewrkcmbs2,geocmbs))
#test
wewrk_metro <- metro

#going to try to get my data into::
      # epsg (SRID):4326 
       #proj4string:+proj=longlat +datum=WGS84 +no_defs
coordinates(wewrk2)<- ~lon+lat
proj4string(wewrk2)<-"+proj=longlat"

coordinates(wewrkcmbs2)<-~lon+lat
proj4string(wewrkcmbs2)<-"+proj=longlat"

wewrk_L93 <- spTransform(wewrk2,CRS("+proj=longlat +datum=WGS84 +no_defs"))
wewrkcmbs2_L93 <-spTransform(wewrkcmbs2,CRS("+proj=longlat +datum=WGS84 +no_defs"))

wewrk_metro <- append_data(wewrk_metro,wewrk3,key.shp = "name",
                           key.data = "city", ignore.duplicates = TRUE,
                           ignore.na = TRUE)

###### Need to layer in census data:
med_inc_statedf <- get_acs(geography = "state",year=2017,variables ="B19019_001",
                           survey="acs5")
#potential ideas (Housing Units - "H001001", 
                # Housing units occupied -"H003003", Interest, 
                #dividens or Net rental income - "B19054_001",
                #GROSS RENT AS A PERCENTAGE OF HOUSEHOLD INCOME IN THE PAST 12 MONTHS - "B25070_001"
              #  population: "B00001_001", household income in past 12 months - B19001_001, 
              # Per Capita Income in past twelve months "B19301_001"
              #Educational attainment by employent status for pop 25-64 "B23006_001"
              #employment status for pop over 16 - "B23025_001")

per_capita_inc <- get_acs(geography = "state",year=2017,variables ="B19301_001",
                          survey="acs5")
population_st <-get_decennial(geography = "state",year="2010",variables =  "P001001")     

###################Non Farm and Small business Loan Data Cleaning #############
sbal<-read_xlsx("sba_doc.xlsx",sheet = "5")
sbal<-data.frame(sbal[4:54,])
sbal<-sbal[,c(-2,-5,-8)]

colnames(sbal)<- c('state','lending_under1m','num_under1m','lending_under100k',
                   'num_under100k','num_employees','sbl_amount_per_emp')
rownames(sbal)<-NULL
sbal$lending_under1m<-Numberize(sbal$lending_under1m)
sbal$num_under1m<-Numberize(sbal$num_under1m)
sbal$lending_under100k<-Numberize(sbal$lending_under100k)
sbal$num_under100k<-Numberize(sbal$num_under100k)
sbal$num_employees<-Numberize(sbal$num_employees)
sbal$sbl_amount_per_emp<-Numberize(sbal$sbl_amount_per_emp)
sbal$num_employees_mil <- sbal$num_employees/1000000


non_farm <- read_xlsx("nonfarm.xlsx")
non_farm <- data.frame(non_farm[,1:5])
non_farm <- non_farm[5:59,]

colnames(non_farm) <- c("state","Feb 2018","Dec 2018","Jan 2019","Feb 2019")
non_farm <-na.omit(non_farm)
rownames(non_farm)<-NULL

non_farm$`Feb 2018` <- Numberize(non_farm$`Feb 2018`)
non_farm$`Dec 2018` <- Numberize(non_farm$`Dec 2018`)
non_farm$`Jan 2019` <- Numberize(non_farm$`Jan 2019`)
non_farm$`Feb 2019` <- Numberize(non_farm$`Feb 2019`)
#non calc growth:
non_farm$growth<- ((non_farm$`Feb 2019`/non_farm$`Feb 2018`)-1)*100

str(non_farm)

#US_state2 houses all data relevant, use this for map making:
#a log of this is appending external data like non_farm, and census api info into a combined dataframe to project on a map
US_state2 <- append_data(US_state2,per_capita_inc,key.shp = "STATE",key.data = "GEOID")
US_state2 <- append_data(US_state2,non_farm,key.shp="NAME",key.data = "state")
US_state2 <- append_data(US_state2,med_inc_statedf,key.shp = "STATE",key.data = "GEOID")
US_state2<-append_data(US_state2,population_st,key.shp = "STATE",key.data = "GEOID")  
US_state3<-append_data(US_state2,sbal,key.shp = "NAME",key.data = "state")
US_state_med_inc<-append_data(US_state2,med_inc_statedf,key.shp = "STATE",key.data = "GEOID")
#need to filter only US Metros to show up on map, as it's showing some mexican and candian cities
US_metro <-filter(metro,iso_a3 %in% "USA")
#also need to filter the populations to largest cities
US_metro <-filter(US_metro,pop2010 > 2300000)
sqft_lab <- c("1 mn","2 mn","3 mn","4 mn","5 mn")
col_lab<-get_brewer_pal("grey",5)
wewrkcmbs2_L93@data$sqft_totk <- wewrkcmbs2_L93@data$sqft_tot/1000

usmap_medinc<-tm_shape(US_state2)+
  tm_fill("grey70")+
  tm_shape(US_state_med_inc)+
  tm_polygons("estimate.data",title= "Median income",legend.show=T)+
  #breaks=c(15000,25000,35000,45000,55000))+ 
  #tm_layout(legend.outside = T, legend.outside.position = "bottom",
  #        legend.format = list(text.align="right", text.to.columns = TRUE))+
  tm_shape(wewrk_L93)+
  tm_bubbles("sqft_tot", scale = 2,title.size="Square footage")+#,legend.size.show=F)+ 
  tm_legend(legend.outside = T)

tmap_save(usmap_medinc_cmbs,"Median_inc_cmbs.jpg")

usmap_percap<-tm_shape(US_state2)+
  tm_fill("grey70")+
  tm_shape(US_state2)+
  tm_polygons("estimate",title= "Per Capita Income",legend.show=T)+
                #breaks=c(15000,25000,35000,45000,55000))+ 
  #tm_layout(legend.outside = T, legend.outside.position = "bottom",
  #        legend.format = list(text.align="right", text.to.columns = TRUE))+
tm_shape(wewrk_L93)+
  tm_bubbles("sqft_tot", scale = 2,title.size="Square Footage")+#,legend.size.show=F)+ 
  tm_legend(legend.outside = T)   #legend.stack="horizontal", #legend.position = "bottom")#legend.format = list(text.align="right", text.to.columns = TRUE))+
 # tm_legend(labels=sqft_lab)
  #tm_add_legend(type = "symbol",col = "grey",border.col="black",size=c(.4,.8,1.2,1.6,2),
   #             shape = 1,labels=sqft_lab,legend.format = list(text.align="right"))
#  tm_legend(legend.just="right")

usmap_medinc_cmbs<-tm_shape(US_state_med_inc)+
  tm_fill("grey70")+
  tm_shape(US_state_med_inc)+
  tm_polygons("estimate.data",title= "Median income",legend.show=T)+
  #breaks=c(15000,25000,35000,45000,55000))+ 
  #tm_layout(legend.outside = T, legend.outside.position = "bottom",
  #        legend.format = list(text.align="right", text.to.columns = TRUE))+
  tm_shape(wewrkcmbs2_L93)+
  tm_bubbles("sqft_tot", scale = 2,title.size="Square footage",col = "black",border.col="grey")+#,legend.size.show=F)+ 
  tm_legend(legend.outside = T)

usmap_pop <-tm_shape(US_state_med_inc)+
  tm_fill("grey70")+
  tm_shape(US_state2)+
  tm_polygons("value",title= "Population",legend.show=T)+
  #breaks=c(15000,25000,35000,45000,55000))+ 
  #tm_layout(legend.outside = T, legend.outside.position = "bottom",
  #        legend.format = list(text.align="right", text.to.columns = TRUE))+
  tm_shape(wewrkcmbs2_L93)+
  tm_bubbles("sqft_tot", scale = 2,title.size="Square footage",col = "black",border.col="grey")+#,legend.size.show=F)+ 
  tm_legend(legend.outside = T)

usmap_nonfarm_cmbs<-1
  tm_shape(US_state2)+
  tm_fill("grey70")+
  tm_shape(US_state2,is.master = T)+
  tm_polygons("growth",title= "Nonfarm Growth Rate (%)",legend.show=T,
              legend.format=list(fun=function(x) paste0(formatC(x, digits=0, format="f"), ".0%")))+
  #breaks=c(15000,25000,35000,45000,55000))+ 
  #tm_layout(legend.outside = T, legend.outside.position = "bottom",
  #        legend.format = list(text.align="right", text.to.columns = TRUE))+
  tm_shape(wewrkcmbs2_L93)+
  tm_bubbles("sqft_tot", scale = 2,title.size="Square footage",col = "black",border.col="grey")+#,legend.size.show=F)+ 
  tm_legend(legend.outside = T)
  
    #tm_shape("World")+
   # tm_fill("grey70")


small_bus<-tm_shape(US_state3)+
    tm_fill("grey70")+
    tm_shape(US_state3)+
    tm_polygons("num_employees_mil",
                title= "Small business \nemployees (mil.)",legend.show=F,
                legend.format=list(fun=function(x) paste0(formatC(x, digits=0, format="f"), " mil.")))+
    tm_shape(wewrkcmbs2_L93)+
    tm_symbols("sqft_totk", scale = 2,title.size="Square footage",col = "black",border.col="grey",
              sizes.legend = c(250,500,750,1000),
              legend.format=list(fun=function(x) paste0(formatC(x, digits=0, format="f"), "K")))+#,legend.size.show=F)+ 
    tm_legend(legend.outside = F)+
    tm_credits("Small business data sourced from the small business loan administration, www.sba.gov",
               position = c("left","bottom"),size = .5)+
    tm_shape(can_mex)+
    tm_polygons(col = "#E4E4E4")
    
  tm_style("natural")
  
can_mex<-World[!World$iso_a3=="USA" ,]  
  
  tm_format("World")
qtm(world)

tm_shape(US_metro)+
 tm_text("name",size = .5, clustering = T)
tmap_save(usmap_percap,"Percap.jpg")



##########################################     NYC Map       ####################################
#get the necessary shape file:
options(pkgdown.internet = FALSE)
regions <- ne_download(scale = "large", type = "states",category = "cultural")
View(regions)

lookup_code("New York", "New York")
lookup_code("New York", "Kings")
lookup_code("New York", "Queens")
lookup_code("New York", "Bronx")
lookup_code("New York", "Richmond?")

#download the tracts:
nyc_tracts <- tracts(state = '36', county = c('061','047','081','005','085'))

#this needs to be ran not on a vpn or comp wifi
r <- GET('http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson')
nyc_neighborhoods <- readOGR(content(r,'text'), 'OGRGeoJSON', verbose = F)

set.seed(42)
lats <- 40.7544882 + rnorm(10)/100
lngs <- -73.9879923 + rnorm(10)/200
points <- data.frame(lat=lats, lng=lngs)
points

#The coordinates function specifies which columns should be used for positioning, and 
#the proj4string function specifies what type of projection should be used.
#want a projection that’s consistent with the neighborhood shapes
#we use the matches function to do the spatial join and bind the columns back together

points_spdf <- points
coordinates(points_spdf) <- ~lng + lat
proj4string(points_spdf) <- proj4string(nyc_neighborhoods)
matches <- over(points_spdf, nyc_neighborhoods)
points <- cbind(points, matches)
points

unique(nyc_neighborhoods$borough)
#remove staten (hard to read map)
manhattan_neigh <- nyc_neighborhoods[nyc_neighborhoods$borough=="Manhattan",]
other_nyc_neigh <- nyc_neighborhoods[!nyc_neighborhoods$borough=="Manhattan",]
nycwewrk <- 
tm_shape(nyc_neigh2)+
  tm_borders()

wewrknyc <-wewrk2
#proj4string(wewrknyc)<-proj4string(manhattan_neigh)
#matches <- over(wewrknyc,manhattan_neigh)
#wewrknyc

wewrkgeo <- geocode(paste(wewrk$Address,wewrk$city,wewrk$State,sep = ", "),source = "dsk")
wewrknyc <- cbind(wewrk,wewrkgeo)

wewrkgeo <-data.frame(lat=wewrkgeo$lat,lng=wewrkgeo$lon)
wewrknyc_points <-wewrkgeo
coordinates(wewrknyc_points)<- ~lng + lat
proj4string(wewrknyc_points)<- proj4string(manhattan_neigh)
matches <- over(wewrknyc_points,manhattan_neigh)
wewrknyc_points<-cbind(wewrknyc,matches)

#omit na:
wewrknyc_points<- na.omit(wewrknyc_points)

#now get points into spatial points DF:
wewrknyc_geo <-wewrknyc_points[,7:8]
wewrknyc_points <- SpatialPointsDataFrame(coords = wewrknyc_geo,data = wewrknyc_points,
                                          proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0 "))

#now to get neighborhood tots:
wewrknyc2 <-wewrknyc_points@data
wewkrnyc3 <-wewrknyc2[,c(4,9)]
wewrknyctot


wewrknyctot <-data.frame(tapply(wewrknyc2$SF.Occupied,wewrknyc2$neighborhood,sum))

#get city names back prepare data to load into geocode function
wewrknyctot$neighborhood <- rownames(wewrknyctot)
rownames(wewrknyctot)=NULL
colnames(wewrknyctot)=c("sqft_tot","neighborhood")
wewrknyctot <- merge(wewrknyctot,wewrknyc_points,"neighborhood")

wewrknyctot2<-wewrknyctot
wewrknyctot2$sqft_totk <- wewrknyctot2$sqft_tot/1000
wewrknyctot

#wewrkgeo <-data.frame(lat=wewrknyctot$lat,lng=wewrknyctot$lon)
#wewrknyc_points <-wewrkgeo
coordinates(wewrknyctot)<- ~lon + lat
proj4string(wewrknyc_points)<- proj4string(manhattan_neigh)

rownames(manhattan_neigh2@data)=NULL
manhattan_neigh2<-manhattan_neigh

#can merge data frame based on bolygon, just need to find the "match" value
manhattan_neigh2@data <- data.frame(manhattan_neigh2@data,
                                    wewrknyctot2[match(manhattan_neigh2@data[,1],wewrknyctot2@data[,1]),])
#converts na values to what R will recognize as "White"
manhattan_neigh2@data$sqft_tot[is.na(manhattan_neigh2@data$sqft_tot)] <- 0


#manhattan_neigh2@data<- na.omit(manhattan_neigh2@data)

matches <- over(wewrknyc_points,manhattan_neigh)
wewrknyc_points<-cbind(wewrknyctot,wewrknyc_points)

yel_org_red <- get_brewer_pal("YlOrRd",7)
#found the following palettes using the get_brewer pal for ylorrd
manhattan_palette <- c("#FFFFFF","#FFF0A8","#FECF6B","#FD9B42","#FC502A","#D30F20")
manhattan_neigh2@data$sqft_totk <- manhattan_neigh2@data$sqft_tot/1000 
#first shape argument with units - is for the scale bar argument, otherwise it would be in KM

manhat_tmap<-
  tm_shape(manhattan_neigh,unit = "miles", unit.size=1609)+
  tm_borders("#9F9F9F")+
  tm_shape(manhattan_neigh2)+
  tm_fill(col = "sqft_totk",palette = manhattan_palette,border.col="black",
          legend.show = T,title = "Square footage",breaks = c(0,10,100,250,500,750,Inf),
          legend.format=list(fun=function(x) paste0(formatC(x, digits=0, format="f"), "K")))+
  tm_borders(col = "#747474")+
  tm_text("neighborhood",size = "sqft_tot",scale = .95 ,clustering = T,legend.size.show = F,col="black")+
  tm_shape(other_nyc_neigh)+
  tm_borders("#9F9F9F")+tm_fill("#E4E4E4")+
  tm_shape(NJ_data)+
    tm_borders("#9F9F9F")+tm_fill("#E4E4E4")+
  tm_style("natural")+
  #tm_compass(type="rose",position = c("right", "bottom"),size = 3) +
  tm_scale_bar(position = c("right", "bottom"))+
  
  tm_layout(main.title = "Figure 1: \nWeWork Exposure In Manhattan",main.title.size = 1)
  #C7C7C7 OR #747474
manhat_tmap

tm_shape(manhattan_neigh)+
  tm_borders("#9F9F9F")+
  tm_shape(manhattan_neigh2)+
  tm_fill(col="sqft_totk",palette=manhattan_palette,border.col="black",breaks = c(0,10,100,250,500,750,Inf),
          legend.format=list(fun=function(x) paste0(formatC(x, digits=0, format="f"), "K")))+
          tm_borders(col = "#747474")
tm_shape(hudson)+
    tm_lines(col = "steelblue", lwd = 4)
  
  
  tm_shape(US)+
    tm_borders()
  tm_shape(rivers) +
  tm_lines("lightblue", lwd = "strokelwd", scale = 1.5,
           legend.lwd.show = FALSE)
 


 tm_add_legend(title="Square footage",type = "fill",   labels = (c("Moderate","Moderate-High","High","Extreme")),
              col = yel_org_red)

 
 
 
 
####################################### Weekly Pub ######################################################   

 #first read in csv file:
MF_ny <- data.frame(read.csv(file = "C:/Users/andy_white/Desktop/Projects/Weekly Pub/NYC_mf.csv",header = T,sep=",",dec="."))

MF_ny$Allocated.Balance.Current<-Numberize(MF_ny$Allocated.Balance.Current)
which(MF_ny$Address =="784, 788 & 792 Columbus Avenue")
MF_ny <- MF_ny[-225:-242,]
MF_ny$Address<- as.character(MF_ny$Address)
MF_ny[17,13] <- as.character("792 Columbus Avenue")
MF_ny[26,13] <- as.character("792 Columbus Avenue")
MF_ny[36,13] <- as.character("792 Columbus Avenue")

MF_ny <- cbind(MF_ny,Geo_MF_ny)
MF_ny <- na.omit(MF_ny)
Geo_MF_ny <- na.omit(Geo_MF_ny)
MF_ny_spdf <- SpatialPointsDataFrame(data=MF_ny,coords = Geo_MF_ny,
                                     proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

MF_overlay <- over(MF_ny_spdf,nyc_neighborhoods)
MF_ny <- cbind(MF_ny,MF_overlay)
MF_ny2 <- na.omit(MF_ny)
MF_ny2_geo <-MF_ny2[,78:79]

MF_ny_spdf2 <- SpatialPointsDataFrame(data=MF_ny2,coords = MF_ny2_geo,
                                      proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

MF_ny3 <-MF_ny_spdf2@data
MF_ny3 <-MF_ny3[,c(8,80)]


MF_ny_tot <-data.frame(tapply(MF_ny3$Allocated.Balance.Current,MF_ny3$neighborhood,sum))

#get city names back prepare data to load into geocode function
MF_ny_tot$neighborhood <- rownames(MF_ny_tot)
rownames(MF_ny_tot)=NULL
colnames(MF_ny_tot)=c("loan_bal","neighborhood")
MF_ny_tot

MF_ny_spdf3 <- merge(MF_ny_tot,MF_ny_spdf2,"neighborhood")

MF_ny_spdf4 <-MF_ny_spdf3
MF_ny_spdf4$loan_bal_mil <- MF_ny_spdf4$loan_bal/1000000

install.packages("spatialEco")
library(spatialEco)

#in order for merge function to work between points/polygons, needed to add duplicateGeoms=T
MF_ny_poly2 <- merge(nyc_neighborhoods_mf,MF_ny_spdf4@data,"neighborhood",duplicateGeoms =T)

MF_ny_poly2@data[is.na(MF_ny_poly2@data)]<-0
#MF_ny_poly2@data <- na.omit(MF_ny_poly2@data)

#wewrkgeo <-data.frame(lat=wewrknyctot$lat,lng=wewrknyctot$lon)
#wewrknyc_points <-wewrkgeo
coordinates(MF_ny_spdf4)<- ~lon + lat
proj4string(MF_ny_spdf4)<- proj4string(nyc_neighborhoods)


rownames(nyc_neighborhoods_mf@data)=NULL
nyc_neighborhoods_mf<-nyc_neighborhoods

#can merge data frame based on bolygon, just need to find the "match" value
nyc_neighborhoods_mf@data <- data.frame(nyc_neighborhoods_mf@data,
                                    MF_ny_spdf4[match(nyc_neighborhoods_mf@data[,1],MF_ny_spdf4@data[,1]),])

#converts na values to what R will recognize as "White"
manhattan_neigh2@data$sqft_tot[is.na(manhattan_neigh2@data$sqft_tot)] <- 0
Geo_MF_ny <- geocode(paste(MF_ny$Address,MF_ny$City,MF_ny$State),source = "dsk")

get_brewer_pal("YlOrRd",7)
palette_MF <- c("#F8F8F8","#FEDD80","#FEC05B","#FD9E43","#FC6E33","#D9131E","#800026")


MF_ny_poly3@data
NJ_data2 <- NJ_data[!(NJ_data$NAME %in% c("Bronx","Kings","Richmond","New York","Queens")),]

nyc_MF <- tm_shape(NJ_data2)+
 tm_borders("black")+tm_fill("#9F9F9F")+
tm_shape(MF_ny_poly2,is.master = T)+
  tm_fill("loan_bal_mil",palette=palette_MF,breaks = c(0,1,25,50,75,100,400,Inf),legend.show = F)+
  tm_borders("black")+
  tm_add_legend(type="fill",labels = c("N/A","0-25 mil.","25-50 mil.","50-75 mil.","75-100 mil.",
                                       "100-400 mil.","+400 mil."), col = palette_MF,title = "Loan Balance",
                is.portrait = T)+
  
  #tm_text("neighborhood",size = "loan_bal_mil",
   #       scale = .9 ,clustering = T ,legend.size.show = F,col="black")+
  tm_credits("Data compiled from trepp.com")+
  
tm_style("natural")

tmap_arrange(nyc_MF,NY_state_mf)

MF_ny_state <-MF_ny[,c(8,19)]

MF_ny_state <- data.frame(tapply(MF_ny_state$Allocated.Balance.Current,MF_ny_state$County,sum))

MF_ny_state$County <- rownames(MF_ny_state)
rownames(MF_ny_state)=NULL
colnames(MF_ny_state)=c("loan_bal","County")
MF_ny_state <- na.omit(MF_ny_state)

MF_ny_state2 <- merge(MF_ny,MF_ny_state,"County")


MF_ny_state2$loan_bal <- MF_ny_state2$loan_bal/1000000

manhattan_neigh2@data$sqft_tot[is.na(manhattan_neigh2@data$sqft_tot)] <- 0

coordinates(MF_ny_state2)<- ~lon + lat
proj4string(MF_ny_state2)<- proj4string(NY_data)
NY_data  
NY_data <- filter(US, STATE %in% "36")
NY_data2 <- append_data(NY_data,MF_ny_state2,key.shp = "NAME",key.data = "County",ignore.duplicates = T)

NY_data2$loan_bal[is.na(NY_data2$loan_bal)] <- 0
NY_data3 <- filter(NY_data2,loan_bal > 0)

NY_state_mf <-tm_shape(can_mex)+
  tm_fill("#9F9F9F")+tm_borders("black")+
tm_shape(US_state2)+
  tm_fill("#9F9F9F")+tm_borders("black")+
tm_shape(NY_data2,is.master = T)+
  tm_fill("white")+
  tm_borders("black")+
tm_shape(NY_data3)+
  tm_fill("loan_bal",palette=palette_MF,breaks = c(0,1,25,50,75,100,400,Inf),legend.show = F)+
  tm_borders("black")+
  
  #tm_text("NAME",size = "CENSUSAREA",scale = .65,clustering = T,legend.size.show = F)+
  tm_style("natural")
  

tmap_arrange(nyc_MF,NY_state_mf)
#tm_compass(type="rose",position = c("right", "bottom"),size = 3) +
  tm_scale_bar(position = c("right", "bottom"))




tm_shape(nyc_neighborhoods)+
   tm_borders("#9F9F9F")

MF_ny_spdf2 <- SpatialPointsDataFrame(data=MF_ny2,coords = MF_ny2_geo,
                                      proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))


tm_shape(nyc_neighborhoods)+
  tm_borders(col = "grey")
  


tm_shape(nyc_neighborhoods)+tm_fill("grey")+tm_shape(MF_ny_spdf2)+tm_bubbles(col = "black")


#############################################################################################

us<-map_data("state")
us$statename <-us$region
state.name<-tolower(state.name)
gg<-ggplot(us,aes(map_id=statename))
# the base map
gg<-gg+geom_map(map=us,fill="white",color="black")+expand_limits(x=us$long,y=us$lat)+coord_map()

gg<-gg+geom_point(data=wewrk2,aes(x=geoCode$lon,y=geoCode$lat,size=sqft_tot),color="#AD655F")

gg

####Now with TMAP:
f <- tempfile()
download.file("http://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_050_00_20m.zip", destfile = f)
unzip(f, exdir = ".")
US <- read_shape("gz_2010_us_050_00_20m.shp")
US2<-readShapeSpatial("gz_2010_us_050_00_20m.shp")
CA_data<-filter(US, STATE %in% "06")

#this is a download of the states files
s <- tempfile()
download.file("http://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_040_00_20m.zip", destfile = f)
unzip(f, exdir = ".")
US_state <- read_shape("gz_2010_us_040_00_20m.shp")
# leave out AK, HI, and PR (state FIPS: 02, 15, and 72)
US <- US[!(US$STATE %in% c("02","15","72")),]  

#for states: leave out ak, hi, PR
US_state <- US_state[!(US_state$STATE %in% c("02","15","72")),]

# append data to shape
US$FIPS <- paste0(US$STATE, US$COUNTY)
US <- append_data(US, med_inc_df, key.shp = "FIPS", key.data = "GEOID")

US_state <- unionSpatialPolygons(US,US$STATE)

tm_basemap("Stamen.Watercolor")+
  tm_shape(World)+ tm_bubbles(size = US_state$CENSUSAREA, col="red")+
  tm_tiles("Stamen.TonerLabels")

data(US_state)

######################      TESTING    #######################
tm_shape(US_state)

data("World")

tm_shape(World) +
  tm_polygons("HPI")

data(World, metro, rivers, land)



tmap_mode("plot")
## tmap mode set to plotting
tm_shape(land) +
  tm_raster("elevation", palette = terrain.colors(10)) +
  tm_shape(World) +
  tm_borders("white", lwd = .5) +
  tm_text("iso_a3", size = "AREA") +
  tm_shape(metro) +
  tm_symbols(col = "red", size = "pop2020", scale = .5) +
  tm_legend(show = FALSE)

qtm(US,fill="white",color="black")


#gg <- gg + geom_map(data=map, map=map,
   #                 aes(x=long, y=lat, map_id=id,group=group),
    #                fill="#ffffff", color="#0e0e0e", size=0.15)


# your bubbles
gg <- gg + geom_point(data=myData, 
                      aes(x=us$long, y=lat, size=pop), color="#AD655F") 
gg <- gg + labs(title="Bubbles")



old<- c("Long Island","Newark","San Jose","Oakland",
        "Dallas/Fort Worth","Orange County","Riverside")
new <- c("New York", "New York","San Francisco","San Francisco",
         "Dallas","Los Angeles","Los Angeles")
msatest$city[msatest$city %in% old] <- new[match(msatest$city,old)]

##### preparing data to plot on tmap ######
msa<-data.frame(read_xlsx(mttMSA,sheet = "MSA & CBSA Lookup"))

#remove uneeded rows
msa <- msa[,c(-2,-3,-5,-6,-8)]

#remove na's from remaining data
msa <- na.omit(msa)
rownames(msa)<-NULL
#rename columns:
colnames(msa)<- c("zip","city.state","city")

#bring in zip data base to "clean zips"
data("zipcode")

#turn first column into number:
msa$zip <- clean.zipcodes(msa$zip)

msa$city <- replace(msa$city,msa$city=="Long Island","New York")
msa$city <- replace(msa$city,msa$city=="Newark","New York")
msa$city <- replace(msa$city,msa$city=="San Jose","San Francisco")
msa$city <- replace(msa$city,msa$city=="Oakland","San Francisco")
msa$city <- replace(msa$city,msa$city=="Dallas/Fort Worth","Dallas")
msa$city <- replace(msa$city,msa$city=="Orange County","Los Angeles")
msa$city <- replace(msa$city,msa$city=="Riverside","Los Angeles")


wewrk2 = merge(wewrk2,msa,by.x="Zip",by.y="zip")

wewrk3 <-wewrk2[,c(-2,-7,-8,-10)]

fire_map2 <-st_read("cmbs_fire.shp")

tm_shape(CA_data)+
  tm_borders()+
  tm_shape(various_CA)+
  tm_bubbles(size = "LN_CURR")+
  tm_shape(metro)+
  tm_text("name",size = .5, clustering = T)

various_CA <- fire_map2[which(fire_map2$LN_ADDR=="Various"),]


CA_data<-filter(US, STATE %in% "06")