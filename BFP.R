#set up
library(datasets)
library(readr)
url <- "https://docs.google.com/spreadsheets/d/e/2PACX-1vQPprnw
        KGsFQla0eq25Fnwb9fhSmsGBxSNU-cly8z-63fBB2PYiGc_-BhiCSJr9
        ouR_LBh7q-PI8ZCR/pub?output=csv"
bodyfat <- read_csv(url)

#Response variable: BodyFat
#Predictor variables: 14

#libraries
library(ggplot2) 
library(ggridges)
library(ggpubr)
library(corrplot)
library(dplyr)
library(tidyverse)
library(plotly)
library(caret)

#EDA
str(bodyfat)
summary(bodyfat)
df <- bodyfat[-c(42,182),]

#scatter plots
pairs(BodyFat~Density+Age+Weight+Height+Neck+Chest+Abdomen+Hip+Thigh
      +Knee+Ankle+Biceps+Forearm+Wrist,data=bodyfat,cex.labels=1.4)

#function to plot histagram and density plots
plot_hist_dens <- function(x, na.rm = TRUE, ...) {
  name <- names(x)
  plot_lst <- vector("list", length = length(name))
  
  for (i in seq_along(name)) {
    p <- ggplot(x,aes_string(x = name[i])) + 
      geom_histogram(aes(y = ..density..),bins = 50,color = "#1877C9",
                     fill = "#1E88E5",alpha=0.5) +
      geom_density(color = "#E4276C", fill = "#D81B60", alpha = 0.3)+
      theme(aspect.ratio=9/16)
    plot_lst[[i]] <- p
  }
  
  cowplot::plot_grid(plotlist = plot_lst, nrow = ceiling(length(name)/2))
}

#plots
plot_hist_dens(df[,1:4])
plot_hist_dens(df[,5:8])
plot_hist_dens(df[,9:12])
plot_hist_dens(df[,13:15])

#m0
m0 <- lm(BodyFat~.,data=df[,-1])
summary(m0)

par(mfrow = c(2, 2))
plot(m0)

#make this example reproducible
set.seed(666)

#create ID column
df$ids <- 1:nrow(df)

#use 80% of dataset as training set and 20% as test set 
train <- df %>% dplyr::sample_frac(0.80)
test  <- dplyr::anti_join(df, train, by = 'ids')

train <- train[,-c(1,16)]
test <- test[,-c(1,16)]

dim(train)
dim(test)

#scatter plots
pairs(BodyFat~Age+Weight+Height+Neck+Chest+Abdomen+Hip+Thigh
      +Knee+Ankle+Biceps+Forearm+Wrist,data=train,cex.labels=1.4)

#heatmap
corrplot.mixed(cor(train), order = 'AOE')

#heatmap 2
df_train <- train[,-c(3,6,9)]
corrplot.mixed(cor(df_train), order = 'AOE')

#scatter plot
pairs(BodyFat~.,data=df_train,cex.labels=1.4)

#m1
m1 <- lm(BodyFat ~ ., data = df_train)
summary(m1)

par(mfrow = c(2, 2))
plot(m1)

#cook's distance
N <- 252
cutoff <- 4/(N-2)
cutoff
cooksD <- cooks.distance(m1)
influential <- cooksD[(cooksD > cutoff)]
influential

#box_cox
library(alr3)
summary(powerTransform(cbind(BodyFat,Age,Height,Neck,Abdomen,Hip,Knee,
                             Ankle,Biceps,Forearm,Wrist)~1,data=df_train))

#transformation
df_train$logNeck <- log(df_train$Neck)
df_train$logAbdomen <- log(df_train$Abdomen)
df_train$newHip <- 1/(df_train$Hip^2)
df_train$newKnee <- 1/(df_train$Knee)
df_train$logBiceps <- log(df_train$Biceps)
df_train$logForearm <- log(df_train$Forearm)
df_train$newWrist <- 1/(df_train$Wrist)

new_dftrain <- df_train[,-c(4,5,6,7,9,10,11)]

#m2
m2<- lm(BodyFat ~ ., data = new_dftrain)
summary(m2)

par(mfrow = c(2, 2))
plot(m2)

#A plot of BodyFat against fitted values with a straight line added
par(mfrow=c(1,1))
plot(m2$fitted.values,new_dftrain$BodyFat,xlab="Fitted Values",ylab="Body Fat")
abline(lsfit(m2$fitted.values,new_dftrain$BodyFat),col="#aa2c2c")

#mmp
library(alr3)
mmps(m2,layout=c(2,4))

#avp
library(car)
avPlots(m2)

#vif
library(car)
vif(m2)
# A number of these variance inflation factors exceed 5, the cut-off often used, and so 
#the associated regression coefficients are poorly estimated due to multicollinearity.

#all subsets
attach(new_dftrain)
#vars <- cbind(Age,Height,Neck,Ankle,Biceps,abdomen_hip,logWeight,
# newChest,newKnee,newForearm,logWrist,logThigh)
vars <- cbind(Age,Height,Ankle,newKnee,logNeck,logAbdomen,newHip,logBiceps,
              logForearm,newWrist)
library(leaps)
all_subsets <- regsubsets(as.matrix(vars),BodyFat,nvmax=10)
rs <- summary(all_subsets)
rs
ls(rs)

par(mfrow=c(1,3))
plot(1:10,rs$adjr2,xlab="Subset Size",ylab="Adjusted R-squared")
plot(1:10,rs$cp,xlab="Subset Size",ylab="Cp")
plot(1:10,rs$bic,xlab="Subset Size",ylab="BIC")

library(car) 
par(mfrow=c(1,3))
subsets(all_subsets,statistic=c("adjr2"),legend = T) 
subsets(all_subsets,statistic=c("cp"),legend = T)
subsets(all_subsets,statistic=c("bic"),legend = T)

data.frame(
  Adj.R2 = which.max(rs$adjr2),
  CP = which.min(rs$cp),
  BIC = which.min(rs$bic))

#stepwise
#Backward elimination based on AIC 
N <- length(m2$residuals)
backAIC <- step(m2,direction="backward", data=new_dftrain)

#Backward elimination based on BIC 
backBIC <- step(m2,direction="backward", data=new_dftrain, k=log(N))

#Forward selection based on AIC 
mint <- lm(BodyFat~1,data=new_dftrain)
forwardAIC <- step(mint,scope=list(lower=~1, 
                                   upper=~Age+Height+Ankle+newKnee+logNeck
                                   +logAbdomen+newHip+logBiceps
                                   +logForearm+newWrist),
                   direction="forward", data=new_dftrain)

#Forward selection based on BIC 

forwardBIC <- step(mint,scope=list(lower=~1, 
                                   upper=~Age+Height+Ankle+newKnee+logNeck
                                   +logAbdomen+newHip+logBiceps
                                   +logForearm+newWrist),
                   direction="forward", data=new_dftrain,k=log(N))

detach(new_dftrain)

#testing set
df_test <- test[,-c(3,6,9)]
df_test$logNeck <- log(df_test$Neck)
df_test$logAbdomen <- log(df_test$Abdomen)
df_test$newHip <- 1/(df_test$Hip^2)
df_test$newKnee <- 1/(df_test$Knee)
df_test$logBiceps <- log(df_test$Biceps)
df_test$logForearm <- log(df_test$Forearm)
df_test$newWrist <- 1/(df_test$Wrist)
new_dftest <- df_test[,-c(4,5,6,7,9,10,11)]

#adjR^2
m2.R2 <- lm(BodyFat~Age+Height+logNeck+logAbdomen+logForearm+newWrist, 
            data=new_dftest)
summary(m2.R2)

par(mfrow = c(2, 2))
plot(m2.R2)

#AIC
m2.AIC.6 <- lm(BodyFat~logForearm+logNeck+newWrist+Age+Height+logAbdomen,
               data=new_dftest)
summary(m2.AIC.6)

par(mfrow = c(2, 2))
plot(m2.AIC.6)

#BIC
m2.BIC.4 <- lm(BodyFat~Age + Height + logAbdomen + newWrist,data=new_dftest)
summary(m2.BIC.4)

par(mfrow = c(2, 2))
plot(m2.BIC.4)

#avp
library(car)
avPlots(m2.BIC.4)

#final model
m2.BIC.4.final <- lm(BodyFat~logAbdomen + newWrist,data=new_dftest)
summary(m2.BIC.4.final)

par(mfrow = c(2, 2))
plot(m2.BIC.4.final)
