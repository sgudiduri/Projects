# To run this file you:
#1. You need to ensure that train.csv is in the scripts folder
#2. Run Joining Environmental Datasets to Housing.ipynb which creates 'PhysicalAndEnvironmental.csv'
#3. You can now run this file. (I left the plot(<model name>) commented out since it can cause weird behavior sometimes)
#4. The output for all future processing of the "chosen variables" is in ProcessedVariablesToFit.csv
#4b. The output is saved in this directory

if (!require(dplyr)) install.packages("dplyr")
if (!require(randomForest)) install.packages("randomForest")
if (!require(car)) install.packages("car")
library(dplyr)
library(randomForest)
library(car)
# Load the combined dataset
physicalAndEnvironmentalDataOriginal <- read.csv("PhysicalAndEnvironmental.csv")

physicalAndEnvironmentalData <- physicalAndEnvironmentalDataOriginal

# Split the data
# Set seed for reproducibility
set.seed(123456789)       
inSampleInd <- sample(c(1:1460), size=floor(1460*.75), replace =F)
# Load testData*
testData <- physicalAndEnvironmentalData[-inSampleInd,]
# Set training data
physicalAndEnvironmentalData <- physicalAndEnvironmentalData[inSampleInd,]

# Handle Empty/Missing Data
physicalAndEnvironmentalData <- physicalAndEnvironmentalData %>% 
  # For Lot Frontage if don't have set to 0 as opposed to NA this is numerical
  transform(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage)) %>%
  # For Garage Year Built set 0 for missing values(maybe set categorical?)
  # transform(GarageYrBlt = ifelse(is.na(GarageYrBlt), 0, GarageYrBlt)) %>%
  # For MasVnrArea set 0 for missing values this is numerical
  transform(MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea)) 

#Filter test data*
testData <- testData %>% 
  # For Lot Frontage if don't have set to 0 as opposed to NA this is numerical
  transform(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage)) %>%
  # For Garage Year Built set 0 for missing values(maybe set categorical?)
  # transform(GarageYrBlt = ifelse(is.na(GarageYrBlt), 0, GarageYrBlt)) %>%
  # For MasVnrArea set 0 for missing values this is numerical
  transform(MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea)) 


# Remove the id column
physicalAndEnvironmentalData <- subset (physicalAndEnvironmentalData, select = -c(Id))

# Remove the id column from test data*
testData <- subset (testData, select = -c(Id))

# Convert some numerical columns to factors where appropriate
physicalAndEnvironmentalData <- physicalAndEnvironmentalData %>% 
  # The year the home was built should be a factor
  transform(YearBuilt = as.factor(YearBuilt)) %>%
  # The year the home was remodeled should be a factor
  transform(YearRemodAdd = as.factor(YearRemodAdd)) %>%
  # The year garage was built should be a factor
  transform(GarageYrBlt = ifelse(is.na(GarageYrBlt), "None", GarageYrBlt)) %>%
  transform(GarageYrBlt = as.factor(GarageYrBlt)) %>%
  # The month sold should be 1 - 12 so turn to a factor
  transform(MoSold = as.factor(MoSold)) %>%
  # The year sold should be a factor
  transform(YrSold = as.factor(YrSold))
  
# Convert some numerical columns to factors where appropriate (for test data)
testData <- testData %>% 
  # The year the home was built should be a factor
  transform(YearBuilt = as.factor(YearBuilt)) %>%
  # The year the home was remodeled should be a factor
  transform(YearRemodAdd = as.factor(YearRemodAdd)) %>%
  # The year garage was built should be a factor
  transform(GarageYrBlt = ifelse(is.na(GarageYrBlt), "None", GarageYrBlt)) %>%
  transform(GarageYrBlt = as.factor(GarageYrBlt)) %>%
  # The month sold should be 1 - 12 so turn to a factor
  transform(MoSold = as.factor(MoSold)) %>%
  # The year sold should be a factor
  transform(YrSold = as.factor(YrSold))

hist(physicalAndEnvironmentalData$SalePrice)
hist(log(physicalAndEnvironmentalData$SalePrice))

#Transform SalePrice to log(SalePrice) of it to better normalize dep. variable
physicalAndEnvironmentalData <- physicalAndEnvironmentalData %>% 
  mutate(SalePrice = log(SalePrice))
testData <- testData %>% 
  mutate(SalePrice = log(SalePrice))

# Create factors out of the char types
counter = 1
columnNames <- colnames(physicalAndEnvironmentalData) 
for (columnType in sapply(physicalAndEnvironmentalData, class)) {
  if (columnType == "character")
  {
    physicalAndEnvironmentalData[, c(columnNames[counter])] <- as.factor(physicalAndEnvironmentalData[, c(columnNames[counter])])
    testData[, c(columnNames[counter])] <- as.factor(testData[, c(columnNames[counter])])
    
  }
  counter = counter + 1
}

physicalAndEnvironmentalDataWithoutScaling <- physicalAndEnvironmentalData

# Now scale numerical/integer data types
counter = 1
columnNames <- colnames(physicalAndEnvironmentalData) 
for (columnType in sapply(physicalAndEnvironmentalData, class)) {
  if (columnType == "integer" || columnType == "numeric")
  {
    minTrain <- min(physicalAndEnvironmentalData[, c(columnNames[counter])])
    maxTrain <- max(physicalAndEnvironmentalData[, c(columnNames[counter])])
    minTest <- min(testData[, c(columnNames[counter])])
    maxTest <- max(testData[, c(columnNames[counter])])
    
    physicalAndEnvironmentalData[, c(columnNames[counter])] <- (physicalAndEnvironmentalData[, c(columnNames[counter])] - minTrain)/ (maxTrain - minTrain)
    testData[, c(columnNames[counter])] <- (testData[, c(columnNames[counter])] - minTest)/ (maxTest - minTest)
    
  }
  counter = counter + 1
}

# Fit Stepwise Regression in order to get the best model (by adding and then removing features)
no_feature_model <- lm(SalePrice ~ 1 , physicalAndEnvironmentalData)
all_feature_model <- lm(SalePrice ~ . , physicalAndEnvironmentalData)

stepwiseModel <- step(no_feature_model, direction="both", scope = formula(all_feature_model), trace = 0)
stepwiseModel$anova
stepwiseModel$coefficients

trimmedSetOfCoefficients <- c("OverallQual", "Neighborhood", "GrLivArea", "BsmtFinType1",
                              "GarageCars", "RoofMatl", "OverallCond", "TotalBsmtSF",
                              "SaleCondition", "CentralAir", "BsmtUnfSF", "Functional",
                              "YearBuilt", "MSZoning", "LotArea", "Condition1",
                              "LandSlope",  "WoodDeckSF", "Heating", "BldgType",
                              "Exterior1st", "GarageYrBlt", "BsmtQual", "BsmtExposure",
                              "ScreenPorch", "GarageType", "Fireplaces", "YrSold",
                              "LotConfig", "RoofStyle", "BsmtCond", "BsmtFullBath",
                              "LowQualFinSF", "X3SsnPorch", "Foundation","SaleType",
                              "KitchenQual", "Condition2", "PavedDrive", "GarageArea",
                              "KitchenAbvGr", "OpenPorchSF", "GarageFinish","PoolArea",
                              "MiscFeature", "Utilities", "BsmtHalfBath", "GarageCond",
                              "GarageQual", "MasVnrArea", "SalePrice")  

# This takes us from 85 to 50 variables
swRFilteredVariables <- physicalAndEnvironmentalData[, trimmedSetOfCoefficients]

# Fit the linear model derived by stepwise regression
linearModelByStepWise <- lm(SalePrice ~., data = swRFilteredVariables)
options(max.print=2000)
summary(linearModelByStepWise)
#plot(linearModelByStepWise)
#Multiple R-squared:  0.9662,	Adjusted R-squared:  0.9497 

# To many features so remove some based on the summary from above
# Remove YearBuilt, BsmtFinType1, "GarageYrBlt"
trimmedSetOfCoefficients <- c("OverallQual", "Neighborhood", "GrLivArea",
                              "GarageCars", "RoofMatl", "OverallCond", "TotalBsmtSF",
                              "SaleCondition", "CentralAir", "BsmtUnfSF", "Functional",
                              "MSZoning", "LotArea", "Condition1",
                              "LandSlope",  "WoodDeckSF", "Heating", "BldgType",
                              "Exterior1st",  "BsmtQual", "BsmtExposure",
                              "ScreenPorch", "GarageType", "Fireplaces", "YrSold",
                              "LotConfig", "RoofStyle", "BsmtCond", "BsmtFullBath",
                              "LowQualFinSF", "X3SsnPorch", "Foundation","SaleType",
                              "KitchenQual", "Condition2", "PavedDrive", "GarageArea",
                              "KitchenAbvGr", "OpenPorchSF", "GarageFinish","PoolArea",
                              "MiscFeature", "Utilities", "BsmtHalfBath", "GarageCond",
                              "GarageQual", "MasVnrArea", "SalePrice")  

# Fit Random Forest Model to get importance scores
rFModel <- randomForest(SalePrice ~ ., data = physicalAndEnvironmentalDataWithoutScaling[, trimmedSetOfCoefficients], importance = TRUE)
varImpPlot(rFModel, type = 1)

# From R Documentation:
#Here are the definitions of the variable importance measures. 
# See here to get importance measures explanation: https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/importance
# This takes us from 47 to 30 variables (order of importance is descending)

randomForestImportanceFactors <- c("GrLivArea", "Neighborhood","TotalBsmtSF", 
                                   "OverallQual", "GarageArea", "LotArea",
                                   "OverallCond","GarageCars", "Fireplaces",
                                   "GarageType", "Exterior1st", "MasVnrArea",
                                   "KitchenQual", "BsmtQual", "BsmtFullBath",
                                   "GarageFinish", "MSZoning", "OpenPorchSF",
                                   "BsmtExposure", "BsmtUnfSF", "BsmtCond",
                                   "BldgType", "Foundation", "WoodDeckSF",
                                   "RoofStyle", "GarageQual", "CentralAir",
                                   "PavedDrive", "SaleType", "KitchenAbvGr",
                                   "SalePrice")

rTreeFilteredVariables <- swRFilteredVariables[,randomForestImportanceFactors]

# Fit a linear model (first attempt)
linearModel <- lm(SalePrice ~ ., data = rTreeFilteredVariables)
options(max.print=2000)
summary(linearModel)
plot(linearModel)

# Find leverage points
cooks <-cooks.distance(linearModel)
which(cooks> 1) 
rTreeFilteredVariables[-c(1299, 1076),]
# points cooks complains about are 1299, 1076
# points remove as outliers 692, 1183, 692

# Refit linear model
rTreeFilteredVariables <- rTreeFilteredVariables[-c(1299, 1076),]
linearModel <- lm(SalePrice ~ ., data = rTreeFilteredVariables)
options(max.print=2000)
summary(linearModel)
#plot(linearModel)

factorsFromRegression <- c("GrLivArea", "OverallQual", "TotalBsmtSF",
                           "LotArea", "OverallCond", "GarageCars",  "KitchenQual",
                           "MSZoning", "BsmtUnfSF", "Foundation", "CentralAir",
                           "SaleType","SalePrice")
# remove BsmtExposure due to BsmtExposureNo not getting fit
# remove BsmtCond due to BsmtExposureNo not getting fit

factorsForVif <-  c("GrLivArea", "OverallQual", "TotalBsmtSF",
                    "LotArea", "OverallCond", "GarageCars",  "KitchenQual",
                    "MSZoning", "BsmtUnfSF", "Foundation", "CentralAir",
                    "SaleType")

linearModeFinalForCheckingVif <- lm(GrLivArea ~ ., data = rTreeFilteredVariables[, factorsForVif])
summary(linearModeFinalForCheckingVif)
# Check that all GVIF^(1/(2*Df)) < 2 (for multilinearity:
vif(linearModeFinalForCheckingVif)
# No multicolinearity

originalDatasetToSave <- physicalAndEnvironmentalDataOriginal[, factorsFromRegression]
write.csv(originalDatasetToSave, "ProcessedVariablesToFit.csv")

# Test log linear model
selectedVariableDf <- rTreeFilteredVariables[, factorsFromRegression]
logLinModel <- lm(SalePrice ~ ., data = selectedVariableDf)
summary(logLinModel)
plot(logLinModel)
# Not bad fit
# Multiple R-squared:  0.9124,	Adjusted R-squared:  0.9099 


inSampleMSE <- sum((predict(logLinModel, selectedVariableDf) - selectedVariableDf[, c("SalePrice")])^2)
inSampleMSE
# In Sample 1.645344

# Average Error per Point
1.645344/1094 #0.001503971

# Attempt at for testData
testDataForTesting <- testData[, factorsFromRegression] %>% filter(SaleType != "Oth")
summary(testDataForTesting)
outSampleMSE <- sum((predict(logLinModel, testDataForTesting) - testDataForTesting[, c("SalePrice")])^2)
outSampleMSE
# MSE 3.97066
3.97066/362 #  0.01096867


# Test Linear - Linear Model
selectedVariableDf <- rTreeFilteredVariables[, factorsFromRegression]
selectedVariableDf <- selectedVariableDf %>% mutate(SalePrice = exp(SalePrice ))
linearModel <- lm(SalePrice ~ ., data = selectedVariableDf)
# Multiple R-squared:  0.9161,	Adjusted R-squared:  0.9139 

summary(linearModel)
plot(linearModel)
inSampleMSE <- sum((predict(linearModel, selectedVariableDf) - selectedVariableDf[, c("SalePrice")])^2)
inSampleMSE
# 4.562414

testDataForTesting <- testData[, factorsFromRegression] %>% filter(SaleType != "Oth") %>%
  mutate(SalePrice = exp(SalePrice ))
outSampleMSE <- sum((predict(linearModel, testDataForTesting) - testDataForTesting[, c("SalePrice")])^2)
outSampleMSE
# 13.96753
