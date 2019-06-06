
# Set working library
projects <- "D:/Dev/Sources/Projects/GitProjects/Predicting-Disease-Spread/"

# Shortcuts to folders of interest
CleanData <- paste0(projects,"/CleanData")
Dictionaries <- paste0(projects,"/Dictionaries")
RawData <- paste0(projects,"/RawData")
RCode <- paste0(projects,"/RCode")
RData <- paste0(projects,"/RData")
Output <- paste0(projects,"/Output")

# Load libraries and install them, if necessary
tmp.library.list <- c("haven", "zoo", "fUnitRoots", "tseries", "urca", "lmtest", "forecast", "data.table", "readxl","reshape", "quantmod", "ggplot2", "reshape2", "plyr","scales", "hts", "fpp2", "lubridate","stargazer","GGally")
for (i in 1:length(tmp.library.list)) {
  if (!tmp.library.list[i] %in% rownames(installed.packages())) {
    install.packages(tmp.library.list[i])
  }
  library(tmp.library.list[i], character.only = TRUE)
}
rm(tmp.library.list)

# =============================================================================
# Import data
# =============================================================================
trainraw <- data.table(read.csv(paste0(RawData,"/dengue_features_train.csv"), sep=',', stringsAsFactors = F))
trainlraw <- data.table(read.csv(paste0(RawData,"/dengue_labels_train.csv"), sep=',', stringsAsFactors = F))
testraw <- data.table(read.csv(paste0(RawData,"/dengue_features_test.csv"), sep=',', stringsAsFactors = F))
submission <- data.table(read.csv(paste0(RawData,"/submission_format.csv"), sep=',', stringsAsFactors = F))

head(trainraw)
head(trainlraw)

dim(trainraw)
dim(trainlraw)

# Merge the features data with the labels data
trainprep <- merge(trainraw, trainlraw, by=c("city", "year", "weekofyear"), all.x = T)
nrow(trainprep) == nrow(trainraw) # if true, we did not accidentally merge many-to-one 

dim(trainprep)

# Omit reanalysis_sat_precip_amt_mm variable because of large amount of missing data
trainprep[, reanalysis_sat_precip_amt_mm := NULL]

# Sort the dataset so it's in a clear order
setkeyv(trainprep, c("city", "year", "weekofyear"))

str(trainprep)

# Only select data for 1 city: San Juan
sjdata <- trainprep[city=="sj"]

# =============================================================================
# Split Train and Test 
# =============================================================================

# Look at the structure of data for SJ
str(sjdata)
head(sjdata)

# Format the date column
sjdata[,week_start_date := as.Date(week_start_date,format = "%Y-%m-%d")]

# Look at date ranges
min(sjdata$week_start_date)
max(sjdata$week_start_date)

# Preserve 5 years of data for model validation
sjdata.train <- sjdata[week_start_date < "2003-01-01"]
sjdata.test <- sjdata[week_start_date >= "2003-01-01"]

# =============================================================================
# Descriptive Analysis & Data Visualization
# =============================================================================

# Train Data
head(sjdata.train)

# Summary
summary(sjdata.train)

# Correlation plots
ggcorr(sjdata.train[, 5:24], label = TRUE, hjust = 0.85, size = 2,color = "grey50",
       label_size = 2) +
  ggplot2::labs(title = "Correlation Plot (San Juan)")

# Plot total cases with timeline -> seasonal effect
ggplot(sjdata.train , aes(week_start_date, total_cases)) + 
  geom_line() + 
  scale_y_continuous() +
  scale_x_date(breaks = date_breaks("year"), labels = date_format("%m-%Y")) +
  ylab("Total cases in San Juan") +
  xlab("")+
  theme(axis.text.x=element_text(angle=-70, hjust=0.001)) +
  labs(title = "Time Series Plot of Total Cases (San Juan)") +
  theme_bw()

# Scatter Plot Total cases vs. Max Temperature
plot(total_cases ~ station_max_temp_c, data=sjdata.train
     ,main = "Scatter Plot of Total Cases and Maximum Temperature in San Juan")

# Scatter Plot Total cases vs. Humidity
plot(total_cases ~ reanalysis_specific_humidity_g_per_kg, data=sjdata.train,
     main = "Scatter Plot of Total Cases and Humidity in San Juan")

# Time Series Plot of log(Total cases) & Max Temperature
ggplot(sjdata.train, aes(week_start_date)) +
  geom_line(aes(y = station_max_temp_c, colour = "station_max_temp_c")) +
  geom_line(aes(y = (y = log(total_cases+1)), 
                colour = "log total_cases")) +
  labs(title = "Time Series Plot of Log Total Cases and Maximum Temperature (San Juan)") +
  theme_bw()

# Time Series Plot of log(Total cases) & Humidity
ggplot(sjdata.train, aes(week_start_date)) +
  geom_line(aes(y = log(total_cases+1), 
                colour = "log total_cases")) +
  geom_line(aes(y = reanalysis_specific_humidity_g_per_kg, 
                colour = "reanalysis_specific _humidity_g_per_kg")) +
  labs(title = "Time Series Plot of Log Total Cases and Humidity (San Juan)") +
  theme_bw()

# Build a function to impute data with the most recent that is non missing (Using LOCF)
na.locf.data.frame <- 
  function(object, ...) replace(object, TRUE, lapply(object, na.locf, ...))

# Fill in NAs
sjdata.train.imputed <- na.locf.data.frame(sjdata.train)
summary(sjdata.train.imputed)

# =============================================================================
# Diagnostics 
# =============================================================================

# Convert data to TS with weekly frequency
sjtrain.ts <- ts(sjdata.train$total_cases, frequency = 52, start = c(1990,18))
head(sjtrain.ts)

# Stationary tests

# ADF Test
adf.test(sjdata.train$total_cases)

# KPSS Test
kpss.test(sjdata.train$total_cases)

# The TS is stationary!

# ACF Plot
ggAcf(sjtrain.ts) +
  labs(title = "Autocorrelation Function Plot (San Juan)") +
  theme_bw()

# PACF Plot
ggPacf(sjtrain.ts)  +
  labs(title = "Partial Autocorrelation Function Plot (San Juan)") +
  theme_bw()

# There's autocorrelation & seasonality!

# Do a decomposition of the TS
decomp = stl(sjtrain.ts, s.window="periodic")
plot(decomp)

# =============================================================================
# Model 1 - ARIMA 
# =============================================================================

# Fit auto.arima without seasonality
sj.fit1 <- auto.arima(sjtrain.ts, seasonal = F)
sj.fit1

checkresiduals(sj.fit1)

# NOTE : Loop exiting after 1 iteration
fit <- list(aicc=Inf)
for(i in 1:25)
{
  fit1 <- auto.arima(sjtrain.ts, xreg=fourier(sjtrain.ts, K=i), seasonal=F)
  if(fit$aicc < fit1$aicc)
    fit1 <- fit
  else break;
}
fit1
i
checkresiduals(fit1)

# =============================================================================
# Model 2 - SARIMA 
# =============================================================================

# Fit auto.arima without seasonality
sj.fit2 <- auto.arima(sjtrain.ts, seasonal = T)
sj.fit2

checkresiduals(sj.fit2)

# NOTE : Loop exiting after 1 iteration
fit <- list(aicc=Inf)
for(i in 1:25)
{
  fit1 <- auto.arima(sjtrain.ts, xreg=fourier(sjtrain.ts, K=i), seasonal=T)
  if(fit$aicc < fit1$aicc)
    fit1 <- fit
  else break;
}
fit1
i
checkresiduals(fit1)

# =============================================================================
# Model 3 SARIMA-X
# =============================================================================
# San Juan 

# prepare xreg
sjtrain.ts1 <- ts(sjdata.train.imputed, 
                  freq=365.25/7, 
                  start=decimal_date(ymd("1990-04-30")))

varlist <- c("precipitation_amt_mm"
             ,"reanalysis_dew_point_temp_k"
             ,"reanalysis_relative_humidity_percent"
             ,"station_avg_temp_c"
             ,"station_diur_temp_rng_c"
             ,"station_max_temp_c"
             ,"station_min_temp_c")

create.tslag <- function(x, dataset) {
  cbind(
    Lag0 = dataset[,x],
    Lag1 = stats::lag(dataset[,x],-1),
    Lag2 = stats::lag(dataset[,x],-2),
    Lag3 = stats::lag(dataset[,x],-3),
    Lag4 = stats::lag(dataset[,x],-4)) %>%
    head(NROW(dataset)) 
}

precipitation <- create.tslag(x = "precipitation_amt_mm"
                              ,sjtrain.ts1)

dew.temp <- create.tslag(x = "reanalysis_dew_point_temp_k"
                         ,sjtrain.ts1)

relative.humidity <- create.tslag(x = "reanalysis_relative_humidity_percent"
                                  ,sjtrain.ts1)

avg.temp <- create.tslag(x = "station_avg_temp_c"
                         ,sjtrain.ts1)

diur.temp <- create.tslag(x = "station_diur_temp_rng_c"
                          ,sjtrain.ts1)

max.temp <- create.tslag(x = "station_max_temp_c"
                         ,sjtrain.ts1)

min.temp <- create.tslag(x = "station_min_temp_c"
                         ,sjtrain.ts1)


# NOTE : Understand this better
# CCF plot -> week 3,4
#differencing 
sjtrain.ts.diff1 = diff(sjtrain.ts,1)
dew.temp.diff1 = diff(dew.temp[,1],1)
#ccf(dew.temp.diff1,
#    sjtrain.ts.diff1,
#    type = c("correlation","covariance")) 

# final model
fw <- fourier(sjtrain.ts, K=2)

fit2 <- auto.arima(sjtrain.ts, xreg=cbind(precipitation[,4], relative.humidity[,4]
                                          ,dew.temp[,4], avg.temp[,4]
                                          ,diur.temp[,4],max.temp[,4]
                                          ,min.temp[,4],fw))

fit2

checkresiduals(fit2)

fit3 <- auto.arima(sjtrain.ts, xreg=cbind(precipitation[,1],relative.humidity[,1]
                                          ,dew.temp[,1],avg.temp[,1]
                                          ,diur.temp[,1],max.temp[,1]
                                          ,min.temp[,1],fw))
fit3
checkresiduals(fit3)


# =============================================================================
# Model 4 - NN
# =============================================================================

# Fit Neural Network
fit4 <- nnetar(sjtrain.ts,xreg=sjtrain.ts1[,5:23])
fit4

# =============================================================================
# Model Selection 
# =============================================================================

# Fill in NAs
sjdata.test.imputed <- na.locf.data.frame(sjdata.test)
summary(sjdata.test.imputed)

# Convert data to TS
sjtest.ts1 <- ts(sjdata.test.imputed, 
                 freq=365.25/7, 
                 start=decimal_date(ymd("2003-01-01")))

# Take lags of predictors
precipitation1 <- create.tslag(x = "precipitation_amt_mm"
                               ,sjtest.ts1)

dew.temp1 <- create.tslag(x = "reanalysis_dew_point_temp_k"
                          ,sjtest.ts1)

relative.humidity1 <- create.tslag(x = "reanalysis_relative_humidity_percent"
                                   ,sjtest.ts1)

avg.temp1 <- create.tslag(x = "station_avg_temp_c"
                          ,sjtest.ts1)

diur.temp1 <- create.tslag(x = "station_diur_temp_rng_c"
                           ,sjtest.ts1)

max.temp1 <- create.tslag(x = "station_max_temp_c"
                          ,sjtest.ts1)

min.temp1 <- create.tslag(x = "station_min_temp_c"
                          ,sjtest.ts1)

# Try ARIMA
fc0 <- forecast(sj.fit1,h=277)
plot(fc0)
accuracy(fc0$mean, as.integer(sjtest.ts1[,"total_cases"]))

# Try SARIMA
fc1 <- forecast(fit1, xreg=fourier(sjtrain.ts, K=1, h=277))
plot(fc1)
accuracy(fc1$mean, as.integer(sjtest.ts1[,"total_cases"]))

# Try SARIMAX with lag 4
fwf <- fourier(sjtrain.ts, K=2, h=277)
fc2 <- forecast(fit2, xreg=cbind(precipitation1[,4],relative.humidity1[,4]
                                 ,dew.temp1[,4],avg.temp1[,4]
                                 ,diur.temp1[,4],max.temp1[,4]
                                 ,min.temp1[,4],fwf), h =277)
plot(fc2)
accuracy(fc2$mean, as.integer(sjtest.ts1[,"total_cases"]))

# Try SARIMAX with lag 1
fc3 <- forecast(fit3, xreg=cbind(precipitation1[,1],relative.humidity1[,1]
                                 ,dew.temp1[,1],avg.temp1[,1]
                                 ,diur.temp1[,1],max.temp1[,1]
                                 ,min.temp1[,1],fwf), h =277)
plot(fc3)
accuracy(fc3$mean,as.integer(sjtest.ts1[,"total_cases"]))


# Try NN
fc4 <- forecast(fit4, xreg=sjtest.ts1[,5:23], h=277)
autoplot(fc4)
accuracy(fc4$mean,as.integer(sjtest.ts1[,"total_cases"]))

