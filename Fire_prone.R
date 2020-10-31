#load the following packages to prepare for data extraction, geospatial analysis, and ultimatly 
#presentation of CMBS loans within high fire danger

###Created date: 1/30/2019

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
library(maps)
library(raster)
library(lwgeom)
library(mailR)
#install.packages("lwgeom")
#install.packages("mailR")
#these datasets allow for more effective color schemes for the maps:
data("World")
data("land")
data("metro")
data("NLD_muni")
data("rivers")
options(scipen = 100000)

#load the following shape maps into a filter
US_metro <-filter(metro,iso_a3 %in% "USA")
Cali_metro_lrg<-US_metro[!(US_metro$name %in% c("Las Vegas", "Riverside","San Jose")),]
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
US_state2 <- read_shape("gz_2010_us_040_00_20m.shp")
# leave out AK, HI, and PR (state FIPS: 02, 15, and 72)
US <- US[!(US$STATE %in% c("02","15","72")),]  
US_state2 <- US_state2[!(US_state2$STATE %in% c("02","15","72")),]

#this is a file provided by california, this shows fire danger levels
fire <-tempfile()
download.file("http://frap.fire.ca.gov/webdata/data/statewide/fhszs.sn.zip",destfile = fire)
unzip(fire,exdir = ".")
fire_map<-read_shape("fhszs06_3.shp")
fire_map2
#fire_map3 <- spTransform(fire_map,CRS("+proj=utm +zone=51 ellps=WGS84"))

yel_org_red <- get_brewer_pal("YlOrRd",4)

#test the fire map to understand what the data looks like
fire_map3 <- fire_map[1:5,]
fire_map2 <-st_read("fhszs06_3.shp") #%>%

#transform into long_lat
fire_map3<-st_transform(fire_map3,"+proj=longlat +ellps=WGS84 +datum=WGS84")  #need to set CRS the same as your dataframe below
fire_map2<-st_transform(fire_map2,4326)

#set the buffer to 1000ft (it's in meters)
#fire_map2<-st_buffer(fire_map,304)
#fire_map3<-st_crs(fire_map,4326)


#must get cmbs into a sf point file so as to find where points intersect with polygons from above
#note the CRS 4326 is the same as fire_map2

#lots of data, so testing was done to make sure that this will work:
d = data.frame(a="YOU DID IT",lon,lat)
dt_sf<-st_as_sf(d,coords = c("lon","lat"),crs=4326,agr = "constant")

#read in local CMBS file of all california properties on our book:
cmbs3<-read.csv("C:/Users/andy_white/Desktop/Projects/Weekly Pub/Weekly Pub/cali_props.csv")

#need to convert the file to sf object, thus the points can be overlay
cmbssf<-st_as_sf(cmbs3,coords = c("lon","lat"),crs=4326,agr = "constant")


#was returning error until I put through st_make_valid. need to find. This will take 15 mins to execute
inter <- st_intersection(cmbssf,st_make_valid(fire_map2)) 

#firemap_density <- smooth_map(fire_map,var="HAZ_CODE",breaks=c(0,1,1.5,2,2.5,3))

st_write(inter,"cmbs_fire.shp")

# now on tto the main event - create the map:

#
CMBS_Fire_map<-tm_shape(cmbs_fire2)+#tm_fill("HAZ_CODE",title="Hazard Class",palette = "YlOrRd",legend.show = F)+
  tm_bubbles(size="LN_CURR_BAL",scale = 2,col = "HAZ_CODE",clustering = T,legend.col.show=F,legend.size.show=F,
             title.size="Loan current balance size")+
  tm_layout(legend.outside = T)+
  tm_shape(CA_data,is.master = T)+
  tm_borders()+
  tm_shape(rivers) +
  tm_lines("lightblue", lwd = "strokelwd", scale = 1.5,
           legend.lwd.show = FALSE) + 
  tm_compass(position = c(0.01, 0.45), color.light = "gray90",
             size = 3) +
  #tm_credits("Projection by Andy White, CFA, created from S&PGR CMBS data and California State Government http://frap.fire.ca.gov",
  #          position = c("LEFT", "BOTTOM"),size = .5)+
  tm_add_legend(title="Fire hazard type",type = "fill",labels = (c("Moderate","Moderate-High","High","Extreme")),
                col = yel_org_red)
# tm_shape(cmbs_fire2)+
#tm_symbols(size = "LN_CURR_BAL",col = "black",scale = 2)

tmap_save(CMBS_Fire_map,"cmbs_fire2.html")

#create some stats, for a write-up
cmbs_haz4<- cmbs_fire2[which(cmbs_fire2$HAZ_CODE=="4"),]

s1 <- sum(cmbs_haz1$LN_CURR_BAL)
s2<-sum(cmbs_haz2$LN_CURR_BAL)
s3<- sum(cmbs_haz3$LN_CURR_BAL)
sum(cmbs_haz4$LN_CURR_BAL)
View(cmbs_fire2)

####DATA Citation USGS 1:100,000 DLGs  / CAL FIRE State Responsibility Areas (SRA05_5)
# CAL FIRE Fire Hazard Severity Zones (FHSZS06_3)

fire_dang<-tm_shape(fire_map)+#tm_fill("HAZ_CODE", title="Hazard Class",palette = "YlOrRd",legend.show = F)+
  tm_add_legend(title="Fire Hazard Type",type = "fill",labels = (c("Moderate","Moderate-High","High","Extreme")),
                 col = yel_org_red)+ #tm_legend(legend.position=c("bottom","left"))
              tm_shape(CA_data,is.master = T)+tm_borders()+
              tm_shape(Cali_metro_lrg)+tm_text("name",size = .5)+
  tm_shape(rivers) +
  tm_lines("lightblue", lwd = "strokelwd", scale = 1.5,
           legend.lwd.show = FALSE)+
  tm_credits("Projection by Andy White, CFA, created from California State Government data http://frap.fire.ca.gov",
             position = c("LEFT", "BOTTOM"),size = .5)



#this will eventually send a mail to folks for distribution
setwd("C:/Users/andy_white/Desktop/Projects/Weekly Pub/Weekly Pub")

send.mail(from = "andy.white@spglobal.com",to="andy.white@spglobal.com",
          subject = "Test",
          body = "//C:/Users/andy_white/Desktop/Projects/Weekly Pub/Weekly Pub/cmbs_fire2.html",
          html = T,
          inline = T,
          smtp = list(host.name = "smtp.office365.com", port = 587,
                      user.name = "andy.white@spglobal.com", passwd = "Mancity6!", tls = TRUE),
          send = T)  

send.mail(from = "andy.white@spglobal.com",to="andy.white@spglobal.com",
          subject = "subject",
          body = "msg", 
          authenticate = TRUE,
          smtp = list(host.name = "smtp.office365.com", port = 587,
                      user.name = "andy.white@spglobal.com", passwd = "Mancity6!", tls = TRUE))



#cmbs_fire<-inter
cmbs_haz4<- cmbs_fire2[which(cmbs_fire2$HAZ_CODE=="4"),]

s1 <- sum(cmbs_haz1$LN_CURR_BAL)
s2<-sum(cmbs_haz2$LN_CURR_BAL)
s3<- sum(cmbs_haz3$LN_CURR_BAL)
sum(cmbs_haz4$LN_CURR_BAL)
View(cmbs_fire2)

####DATA Citation USGS 1:100,000 DLGs  / CAL FIRE State Responsibility Areas (SRA05_5)
# CAL FIRE Fire Hazard Severity Zones (FHSZS06_3)

cmbs3<-read.csv("C:/Users/andy_white/Desktop/Projects/Weekly Pub/Weekly Pub/cali_props.csv")
cmbs4<-cmbs3[1:2,]
#cmbs.sp<-as.data.frame(cmbs4)
#cmbs.sp <- SpatialPointsDataFrame(c(cmbs.sp[,c('lon','lat')]),data = cmbs.sp)
#project points in AEA format
#mapproject(cmbs.sp$lon,cmbs.sp$lat,projection ="albers",c(20,50))

#Load in Trep CMBS DATA:
treppfile <- "C:/Users/andy_white/Desktop/Projects/Weekly Pub/CA_TREPP_Loan_Level_Extract_20180820.xlsx"
cmbs <-read_xlsx(treppfile,sheet = "Sheet3")
bal_update <- read_xlsx(treppfile,sheet = "Sheet2")

#function removed too many dups. Will come back at a later junction
#df.unique <- cmbs[!duplicated(cmbs$LN_PROPNAME), ]
#View(df.unique)

cmbs$add_city_state <- paste(cmbs$LN_ADDRESS,cmbs$LN_CITY,cmbs$LN_STATE,sep = ", ")



#sub(".*? (.+)", "\\1", D$name)

cmbs$add_city_state<-gsub("^[^&]+&","",cmbs$add_city_state)
cmbs$add_city_state<-gsub("^[^/]+/","",cmbs$add_city_state)
cmbs$add_city_state<-gsub("^[^-]+-","",cmbs$add_city_state)
cmbs$add_city_state <- sub(".*? and(.+)","\\1",cmbs$add_city_state)
cmbs$add_city_state<-gsub("^[^&]+&","",cmbs$add_city_state)
#The pattern is looking for any character zero or more times (.*) up until the first space, and then capturing the one or more characters ((.+)) after that first space. 
#The ? after .* makes it "lazy" rather than "greedy" and is what makes it stop at the first space found. 
#So, the .*? matches everything before the first space, the space matches the first space found, and the capture group ((.+)) matches what we want to keep. 
#Then, the second argument to sub refers back to the capture grouped using \\1 to to replace the entire match
cmbs1<-cmbs[1:2000,]
cmbs2<-cmbs[2001:nrow(cmbs),]
cmbsgeo <- geocode(cmbs1$add_city_state,source="dsk")
cmbsgeo2<- geocode(cmbs2$add_city_state,source="dsk")

cmbsgeo3<-rbind(cmbsgeo,cmbsgeo2)

#Now add the columns back to the data frame to prepare for spatial DF:
cmbs2<-cbind(cmbs,cmbsgeo3)

#We should have around 40 N/A values that we need to remove or try to fix:
cmbsna <-is.na(cmbs2$lat)
cmbs3<-cmbs2
#creates an NA dataframe to redo
cmbsna<- cmbs2[which(is.na(cmbs2$lat)),]
#removes values in lat/lon column that are na
cmbs3<-cmbs3[-which(is.na(cmbs2$lat)),]  

#recreates using only city/state this time:
cmbsna$add_city_state <- paste(cmbsna$LN_CITY,cmbsna$LN_STATE,sep = ", ")  
cmbsnageo <- geocode(cmbsna$add_city_state,source = "dsk")
#remove old lat/lon column
cmbsna<-cmbsna[,-13:-14]
#add in new lat/lon column
cmbsna <- cbind(cmbsna,cmbsnageo)
#

cmbs3 <- rbind(cmbs3,cmbsna)
cmbs5<-cmbs3
#write.csv(cmbs3,file = "cali_props.csv")

#probably the closest I've come:
cmbs3<-cmbs5
fire_map2 <- as(fire_map,'Spatial')
points<-data.frame(lon=cmbs3$lon,lat=cmbs3$lat)
coordinates(points)<- ~lon+lat
proj4string(points)<-proj4string(fire_map2)
matches<-over(points,fire_map2)

#Now we need to create a spatial points df to append into our cali shape
coordinates(cmbs3)<-~lon+lat
proj4string(cmbs3) <- CRS(p4s)
res<-spTransform(cmbs3,CRS("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80"))




cmbs3<-st_as_sf(cmbs3,crs=st_crs(fire_map))
cmbs3 <-crop_shape(CA_data,cmbs3,polygon = F)

#cmbssf <- st_as_sf(cmbs3,coords = c('lon','lat'),crs=st_crs(3857))

#cmbssf %>%
#  sf::st_intersection(fire_map)   #%>%

#cmbs_inter <- st_intersection(cmbs4,fire_map2) 

#cmbs3<-st_transform(cmbs3,crs = 4269)
#cmbs3<-st_as_sf(cmbs3,crs=4269)

point_count = lengths(st_contains(cmbs4,fire_map2))
# dplyr::add_count(HAZ_CODE)
#cmbs_latlon <- coordinates(cmbs3)



#Creates a point map, now we will try to overlay
cmbs3<-st_set_crs(cmbs3,3857)
#st_coordinates(cmbs3)
cmbs3<-st_sf(cmbs3)

qtm(cmbs3)+qtm(World)
#set projection of the SpatialPointsDataFrame using the projection of the shapefile


#st_crs(cmbs3)<-p4s

########converts long/lat to right cooordinate system
#proj4string(cmbs3)<-CRS("+proj=longlat")
#transforms coordinates to ellips code and creates matching values with fire_map: 
proj4string(cmbs3)<-CRS("+proj=longlat +datum=WGS84")
cmbs3 <- spTransform(cmbs3, CRS("+proj=utm +zone=51 ellps=WGS84"))

#fire_map is a simple feature data frame need to convert our data to this, and then match
cmbs4<-st_as_sf(cmbs3,precision=0)
#cmbs4<-cbind(cmbs3,cmbs_latlon)
#cmbs4$geom<-st_as_sfc(cmbs3)
#cmbs3<-st_as_sfc(cmbs3)
cmbs4<-st_set_crs(cmbs4,4326)


p4s <- "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
cmbs3<-st_set_crs(cmbs3,4269)
fire_map3<-st_set_crs(fire_map,4269)
#join together - works as left join where first keeps all data
cmbs_fire <- st_join(cmbs3,fire_map3)
inters <- st_intersection(st_geometry(cmbs3),st_geometry(fire_map3))

#first 5 rows of fire:
fire5 <- fire_map[5,]
fire_1 <- filter(fire_map, HAZ_CODE %in% "1")


#cmbs3 <- spTransform(cmbs3,proj4string(fire_map))
#fire_map2<-as(fire_map,'Spatial')

#proj4string(fire_map)<-CRS("+proj=longlat")

#proj4string(cmbs3)<-proj4string(fire_map)
#cmbs4<-set_projection(cmbs3,current.projection = "longlat")

#now to crop only the cali points as some address are still coming in incorrect:
cali_cmbs <- crop_shape(cmbs4,CA_data,polygon = T)
qtm(cali_cmbs)

cmbs_expose <- append_data(fire_map2,cali_cmbs,ignore.na = T,ignore.duplicates = T, key.shp = "geometry",key.data = "geometry",ignore.duplicates = T)


####################################     OLD CODE       ################################################# 

#function above worked for 88/100, need to introduce logic to deal with 
test_cmbs <-cmbs[1:2,]

#this will remove all "&", but i think i want to keep anything after the first
test_cmbs<-gsub(".*&","",test_cmbs$add_city_state)

#Revoe the first "&":
test_cmbs$add_city_state <- gsub("^[^&]+&","",test_cmbs$add_city_state)
test_cmbs<-gsub("^[^-]+-","",test_cmbs$add_city_state)
#test_cmbs<-gsub("^[^and ]+and \\s*","",test_cmbs$add_city_state)

#Cutting up the DF to run geocode function
cmbs1 <- cmbs[1:500,]
cmbs2 <-cmbs[2001:nrow(cmbs),]
#cmbs1_geo <-geocode(cmbs1$add_city_state,source = "dsk")
cmbs4_geo <-geocode(cmbs2$add_city_state,source = "dsk")

cmbs_geo<-rbind(cmbs_geo,cmbs2_geo)
cmbs_geo3 <-rbind(cmbs_geo,cmbs2_geo)
cmbsgeo <- rbind(cmbs_geo3,cmbs4_geo)

x = st_sf(a = 1:3, geom = st_sfc(st_point(c(1,1)), st_point(c(2,2)), st_point(c(3,3))))
y = st_buffer(x, 0.1)
x = x[1:2,]
y = y[2:3,]
plot(st_geometry(x), xlim = c(.5, 3.5))
plot(st_geometry(y), add = TRUE)
st_join(x,y)

st_crs(pt1,pt2)

data("nz")
data("nz_height")

canterbury = nz %>% filter(REGC2017_NAME == "Canterbury Region")
christchruch<- data.frame(pt1,pt2)
pt1 <- 172.6362
pt2<- -43.5321
coordinates(christchruch)<-~pt1+pt2
st_as_sf(christchruch,crs=2193)


point <- tibble(christchruch)%>%
  st_as_sf(coords = c("pt1", "pt2"), crs = 2193)

st_point(c(pt1,pt2))
st_crs(pt1,pt2)
