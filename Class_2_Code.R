

###### Class 2 Code ####################
###### 2/20/2020 ######################

#install the following packages and 

#install.packages("ggplot2")
library(ggplot2)
#install.packages("ggmap")
library(ggmap)
install_version("zipcode",version = "1.0")
library(zipcode)
#corresponding latitude and longitude for each zip code. 
data("zipcode")
install.packages("readxl")
library(readxl)
options(scipen = 1000000)

#sometimes excel docs have a bunch of tabs of data you don't want. use sheet = "xxxx" in the read_excel function
######## PLEASE UPDATE WITH YOUR FILE LOCATION ######################################
default <- read_excel("C:/Users/andy_white/Desktop/Projects/R/SP_course/SP_Training/CMBS_Practice_Data.xlsx",
                       sheet = "default",col_names = T)
#View the dataframe:
View(default)
str(default)
#call the following columns. How did i know to call the columns? I took a look at the data in excel and in the view
#
default2 <- data.frame(Loan_stat = default$`Loan status at maturity`,Vintage=default$VINTAGE, Prop_type=default$PROPERTY_TYPE,Appraised=as.numeric(default$APPRD_VAL_AMT),Loss_sev=as.numeric(default$LOSS_AMT),
                       DY=as.numeric(default$`Debt Yield for Most recent available period`),Orig_bal=as.numeric(default$LOAN_ORIG_BAL_AMT),
                       City=default$CITY,State=default$ST_CD)
default2
#fun little function
colnames(default2)
default2[1:10,1:8]
default5 <- default2[,c(7,8)]
default5 <- default2[,5:8]
summary(default2)
str(default2)
#Ok now a little bit of that cleaning
#this will remove all values in the DY column that have "No DY"
default2[(default2$DY=="NO DY"),]
default2 <- default2[!(default2$DY=="NO DY"),]

#Calculate a loan to value: simply take the loan appount vector and divide by appraisal amount vector

default2$LTV <- default2$Orig_bal/default2$Appraised
#default2$whateverIWANT <- default2$Orig_bal/default2$Appraised
#Things tend to go a little haywire in the viz if you have N/A's there are multiple strategies,
# but for this purpose just remove. The ! does this. Also sometimes the - will also, but generally is
# for removing string types. N/A is actually not a string in R. 

summary(default2)
default2 <- default2[!is.na(default2$DY),]
default2 <- default2[!is.na(default2$Appraised),]
default2 <- default2[!is.na(default2$Loss_sev),]
rownames(default2) <- NULL
default2

default3 <-  na.omit(default2)


str(default2)
summary(default2)

plot(default2)
str(default2)
####
?ggplot
####
ggplot(default2,aes(y=LTV,x=DY))
default2[1,]

point<- ggplot(default2, aes(y=LTV,x=DY))

point+geom_point()

#LAME
#Lets get some color in the house

point + geom_point(aes(colour = factor(Loan_stat)))

ggplot(default2,aes(y=LTV,x=DY))+geom_point(aes(colour=factor(Loan_stat)))
?geom_point
#lets resize some stuff:
point + geom_point(aes(colour = factor(Loan_stat),size=Orig_bal))

#lets make them a little more transparent
point + geom_point(aes(colour = factor(Loan_stat),size=Orig_bal,alpha=2))
#Bro can't see that

point + geom_point(aes(colour = factor(Loan_stat),size=Orig_bal,alpha=2))+
  ylim(0,10)+xlim(-.15,.2)

#Put a title on that. There we go
point + geom_point(aes(colour = factor(Loan_stat),size=Orig_bal,alpha=2))+
  ylim(0,10)+xlim(-.15,.2) + ggtitle("LTV vs Debt Yield")


# a box plot: 
ggplot(default2, aes(y=DY,fill=Loan_stat))+geom_boxplot()

# A crazy box plot with formatting and junk

ggplot(default2, aes(y=DY,fill = Loan_stat))+
  geom_boxplot() +   #let ggplot know you are using boxplot
  ylim(-.05,.2)+     #limit your access to debt yields of -.05 to .2
  ylab("Debt Yield")+    #R is ordered: so this appears a little backwards but the coord flip function below. Remember ylab/xlab/xlim/ylim are based on the default graph
   ggtitle("Loan Default Type Vs. Debt Yield")+       #Graph title
  labs(fill="Default Type")+      #Legend title
  coord_flip() +              #flips the boxblot from vertical (default) to horizontal
  theme(axis.text.y = element_blank())   ###This removes the tick marks and Y axis labels, notice that the coord flip has been utilized and now this "resets" the axis. 

#### Minor tweak to the title:
ggplot(default2, aes(y=DY,fill = Loan_stat))+
  geom_boxplot() +
  ylim(-.05,.2)+
  ylab("Debt Yield")+
  ggtitle("Loan Default Type \nVs. Debt Yield")+       #Graph title \n lets r know to break title to next line
  labs(fill="Default Type")+      #Legend title
  coord_flip() +              #flips the boxblot from vertical (default) to horizontal
  theme(axis.text.y = element_blank()) 

#################################################################################################
#ok complicated bar chart - a lot of stuff happening to format. Generally getting the data on a chart is easy
# the hard part is always the formatting. LOTS OF GOOGLING WAS USED TO MAKE THE FOLLOWING

#Similar setup - call your dataset, inform ggplot of your axis's
bargraph<-ggplot(default2, aes(y=Loss_sev,x=Vintage,fill=Loan_stat))+   # iwant the graph to differentiate loss severity from maturity default and term, by using fill=loan_stat
  geom_col()
bargraph

#now lets make it "pretty" by adding in x axis labels that correspond to data
bargraph <- bargraph + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +   #this adjusts the text angle to 45 degree angle
  scale_x_continuous(breaks = default2$Vintage)  #for some reason scale_x_continuous requires you to call the dataframe and column
  
### Next, looks like our y axis scale is not great, we need to load a library called scales to modify
library(scales) # if you don't have scales: install.packages("scales")

#Now we need to slightly modify our code from above divide loss severity axis by a billion:  
#repasting:
bargraph<-ggplot(default2, aes(y=Loss_sev/1000000000,x=Vintage,fill=Loan_stat))+
  geom_col()+theme(axis.text.x = element_text(angle = 45, hjust = 1)) +   #this adjusts the text angle to 45 degree angle
  scale_x_continuous(breaks = default2$Vintage)
  
#Now comes the formatting intesive activity. Lets make sure the labels are in dollar_format (part of scales package)
bargraph <- bargraph+scale_y_continuous(labels = dollar_format(accuracy = 0.1),   #accuracy is how many decimal points you want
                            breaks = seq(0,3,.5))+   # breaks signals how many times do you want the axis broken up 
                                                    #for ours I used a nifty function called seq (sequence) it creates a sequence of numbers, so above it would
                                                    #create a number sequence starting at 0 and going up to 3, broken up by .5 so: (0, .5, 1, 1.5, 2, 2.5, 3)
                      labs(x="Vintage",y="Loss severity (billions)",
                           fill="Default Type") #if i had used colour above, then to modify legend I would use colour = "Default Type"
#this will save to file of your project path
ggsave("cool_bar_graph.png",plot = bargraph)




###################################################################################################
###################################################################################################
####################### PlotLy ###################################################

#these are pretty advanced charts, but interactive feature and professional look make them second to none
#may need to download other packages like 'stats', 'graphics', and 'ggmap'
install.packages("plotly")
library(plotly)
install.packages("dplyr")
library(dplyr)


#Update the color type the 
plot_ly(data = default2,
        x=default2$Vintage,
        y= default2$Loss_sev,
        color = ~Loss_sev,       #notice if you use the tilda you don't need to call 
        name = "Simple Plotly Bar",
        type = "bar"
         )


#however the data indicates that 2006 and 2007 are generally higher due to a number of large loans with high loss severities
#lets add some labels to show what these are:
p <- plot_ly(data = default2,
        x=default2$Vintage,
        y= default2$Loss_sev,
        color = ~Loss_sev,       #notice if you use the tilda you don't need to call 
        name = "Term",
        type = "bar",
        text = paste("City: ",default2$City,
                     "<br>State: ",default2$State,   #<br> is break in plotly and allows the popup to show a break- you may have seen this in html code
                     "<br>Appr Value: $", format(default2$Appraised/1000000,digits = 2,nsmall = 1),
                     "<br>Loss Sev: $", format(default2$Loss_sev/1000000,digits = 2,nsmall = 1)),
        hoverinfo = 'text')   #need this to get rid of the default label
(p <- p %>%
  layout(yaxis = list(title="Loss Severity (Billions)"),xaxis=list(title = "Vintage")))

##### Now on to a scatterplot with shading by property type
head(default2)
summary(default2$Prop_type)
plot_ly(data = default2,
        type = "scatter",  #select scatter
        x=~DY,
        y=~LTV,
        
        text = paste("Prop Type: ",default2$Prop_type,  #update the text folks see when they hover
                     "<br>State: ",default2$State,   #<br> is break in plotly and allows the popup to show a break- you may have seen this in html code
                     "<br>Appr Value: $", format(default2$Appraised/1000000,digits = 2,nsmall = 1),
                     "<br>Loss Sev: $", format(default2$Loss_sev/1000000,digits = 2,nsmall = 1)),
        hoverinfo = 'text',
        mode = "markers",
        transforms = list(
          list(type='groupby',
               groups=default2$Prop_type,
               styles = list(
                 list(target="OF", value=list(marker = list(color="red"))),  # a lot of copy and pasting, but broken out by prop type
                 list(target="RT", value=list(marker = list(color="blue"))),
                 list(target="LO", value=list(marker = list(color="green"))),
                 list(target="IN", value=list(marker = list(color="grey"))),
                 list(target="MF", value=list(marker = list(color="yellow"))),
                 list(target="MU", value=list(marker = list(color="black"))),
                 list(target="MH", value=list(marker = list(color="brown")))
               ))
        )
        )

#Plot_ly is a little difficult to manipulate, so need to do with our data
#Check out the tappy function:
prop_type_loss_sev <- tapply(default2$Loss_sev, list(default2$Vintage,default2$Prop_type), sum)
prop_type_loss_sev
#it's taking the sum of loss severity and aggregating by vintage and property type
#remove first two rows (little meaning)
(prop_type_loss_sev<- prop_type_loss_sev[-1:-2,-1])

#last but not least a grouped pie chart with data from 2004-2007
t <- plot_ly() %>%
  add_pie(data = prop_type_loss_sev,labels = colnames(prop_type_loss_sev), values = prop_type_loss_sev[4,],
          name ="2004",domain = list(x=c(0,0.4),y = c(0.55, 1))) %>%
  add_pie(data = prop_type_loss_sev,labels = colnames(prop_type_loss_sev), values = prop_type_loss_sev[5,],
          name ="2005",domain = list(x=c(0.6,1),y = c(0.55, 1))) %>%
  add_pie(data = prop_type_loss_sev,labels = colnames(prop_type_loss_sev), values = prop_type_loss_sev[6,],
          name ="2006",domain = list(x=c(0,0.4),y = c(0, .45))) %>%
  add_pie(data = prop_type_loss_sev,labels = colnames(prop_type_loss_sev), values = prop_type_loss_sev[7,],
          name ="2007",domain = list(x=c(0.6,1),y = c(0, 0.45))) %>%
  layout(title = "Loss Severity Of Property Type <br>By Vintage", showlegend = F,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         annotations = list(
           list(x=0.17,y=1.05,text="2004", showarrow=F),
           list(x=0.83,y=1.05,text="2005", showarrow=F),
           list(x=0.17,y=0.5,text="2006", showarrow=F),
           list(x=0.83,y=0.5,text="2007", showarrow=F)
         )) 
t  

#Save to an HTML file - this will save where you created your project
htmlwidgets::saveWidget(t, "test.html")









