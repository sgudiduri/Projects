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

setwd("C:\\Users\\gatech\\Documents\\Reena Gatech\\MGT 6203")
# Load the combined dataset
physicalAndEnvironmentalDataOriginal <- read.csv("PhysicalAndEnvironmental.csv")
physicalAndEnvironmentalData <- physicalAndEnvironmentalDataOriginal

# Handle Empty/Missing Data
physicalAndEnvironmentalData <- physicalAndEnvironmentalData %>% 
  # For Lot Frontage if don't have set to 0 as opposed to NA this is numerical
  transform(LotFrontage = ifelse(is.na(LotFrontage), 0, LotFrontage)) %>%
  # For Garage Year Built set 0 for missing values(maybe set categorical?)
  # transform(GarageYrBlt = ifelse(is.na(GarageYrBlt), 0, GarageYrBlt)) %>%
  # For MasVnrArea set 0 for missing values this is numerical
  transform(MasVnrArea = ifelse(is.na(MasVnrArea), 0, MasVnrArea)) 


# Remove the id column
physicalAndEnvironmentalData <- subset (physicalAndEnvironmentalData, select = -c(Id))

# Since the dependent variable is skewed, we will take log transformation of sale price

par(mar=c(1,1,1,1))
barplot(hist(physicalAndEnvironmentalData$SalePrice))

barplot(hist(log(physicalAndEnvironmentalData$SalePrice)))


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

# Create factors out of the char types
counter = 1
columnNames <- colnames(physicalAndEnvironmentalData) 
for (columnType in sapply(physicalAndEnvironmentalData, class)) {
  if (columnType == "character")
  {
    physicalAndEnvironmentalData[, c(columnNames[counter])] <- as.factor(physicalAndEnvironmentalData[, c(columnNames[counter])])
  }
  counter = counter + 1
}

physicalAndEnvironmentalDataWithoutScaling <- physicalAndEnvironmentalData
# Now scale numerical/integer data types
#normalize <- function(x) {
 # return ((x - min(x)) / (max(x) - min(x)))
#}
counter = 1
                                                                  
                                                                      
columnNames <- colnames(physicalAndEnvironmentalData) 
for (columnType in sapply(physicalAndEnvironmentalData, class)) {
  if (columnType == "integer" || columnType == "numeric")
 
  {
  
    physicalAndEnvironmentalData[, c(columnNames[counter])] <- scales::rescale(physicalAndEnvironmentalData[, c(columnNames[counter])],to=c(0,1))

  }
  counter = counter + 1
}

# Fit Stepwise Regression in order to get the best model (by adding and then removing features)
no_feature_model <- lm(SalePrice ~ 1 , physicalAndEnvironmentalData)
all_feature_model <- lm(SalePrice ~ . , physicalAndEnvironmentalData)

stepwiseModel <- step(no_feature_model, direction="both", scope = formula(all_feature_model), trace = 0)
stepwiseModel$anova
stepwiseModel$coefficients

trimmedSetOfCoefficients <- c("OverallQual", "GrLivArea", "Neighborhood", "BsmtQual",
                              "RoofMatl", "BsmtFinSF1", "MSSubClass", "BsmtExposure",
                              "KitchenQual", "Condition2", "SaleCondition", "OverallCond",
                              "GarageArea", "LotArea", "ExterQual", "PoolQC", "TotalBsmtSF",
                              "Functional", "Condition1", "KitchenAbvGr", "MasVnrArea",
                              "Exterior1st", "PoolArea", "LotConfig", "LowQualFinSF", "HeatingQC",
                              "ScreenPorch", "LandContour", "Street", "LandSlope", "GarageQual",
                              "GarageCond", "MSZoning", "GarageCars", "WoodDeckSF", "BedroomAbvGr",
                              "TotRmsAbvGrd", "X1stFlrSF", "Foundation", "Fence", "BsmtFullBath",
                              "Fireplaces", "FullBath", "GarageFinish", "HalfBath", "X3SsnPorch", "SalePrice")  

# This takes us from 85 to 47 variables
swRFilteredVariables <- physicalAndEnvironmentalData[, trimmedSetOfCoefficients]

# Fit the linear model derived by stepwise regression
linearModelByStepWise <- lm(SalePrice ~., data = swRFilteredVariables)
summary(linearModelByStepWise)
#plot(linearModelByStepWise)
# Multiple R-squared:  0.9261,	Adjusted R-squared:  0.9177 

# Fit Random Forest Model to get importance scores
rFModel <- randomForest(SalePrice ~ ., data = physicalAndEnvironmentalDataWithoutScaling[, trimmedSetOfCoefficients], importance = TRUE)
varImpPlot(rFModel, type = 1)

# From R Documentation:
#Here are the definitions of the variable importance measures. 
# See here to get importance measures explanation: https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/importance
# This takes us from 47 to 30 variables (order of importance is descending)

randomForestImportanceFactors <- c("GrLivArea", "Neighborhood", "OverallQual",
                                   "TotalBsmtSF", "X1stFlrSF", "LotArea", 
                                   "GarageArea", "GarageCars", "BsmtFinSF1",
                                   "ExterQual", "MSZoning", "MSSubClass",
                                   "KitchenQual", "BsmtQual", "GarageFinish",
                                   "Fireplaces", "Exterior1st", "HalfBath",
                                   "FullBath", "OverallCond", "TotRmsAbvGrd",
                                   "HeatingQC", "BedroomAbvGr", "KitchenAbvGr",
                                   "Foundation", "MasVnrArea", "GarageQual",
                                   "BsmtExposure", "BsmtFullBath", "Condition1"
                                   ,"SalePrice")

rTreeFilteredVariables <- swRFilteredVariables[,randomForestImportanceFactors]

# Fit a linear model (first attempt)
linearModel <- lm(SalePrice ~ ., data = rTreeFilteredVariables)
options(max.print=2000)
summary(linearModel)
#plot(linearModel)

# Find leverage points
cooks <-cooks.distance(linearModel)
which(cooks> 1) 
rTreeFilteredVariables[-c(1299),]
# points cooks complains about are 1299
# points remove as outliers 524, 1183, 692

# Refit linear model
rTreeFilteredVariables <- rTreeFilteredVariables[-c(1299, 524, 1183, 692),]
linearModel <- lm(SalePrice ~ ., data = rTreeFilteredVariables)
options(max.print=2000)
summary(linearModel)
#plot(linearModel)


factorsFromRegression <- c("GrLivArea", "Neighborhood", "OverallQual", "TotalBsmtSF",
                           "LotArea", "BsmtFinSF1", "ExterQual", "MSZoning", "MSSubClass",
                           "KitchenQual", "OverallCond", "KitchenAbvGr", "BedroomAbvGr",
                           "MasVnrArea", "GarageQual", "SalePrice")

factorsForVif <- c("GrLivArea", "Neighborhood", "OverallQual", "TotalBsmtSF",
                   "LotArea", "BsmtFinSF1", "ExterQual", "MSZoning", "MSSubClass",
                   "KitchenQual", "OverallCond", "KitchenAbvGr", "BedroomAbvGr",
                   "MasVnrArea", "GarageQual")

linearModeFinalForCheckingVif <- lm(GrLivArea ~ ., data = rTreeFilteredVariables[, factorsForVif])
summary(linearModeFinalForCheckingVif)
# Check that all GVIF^(1/(2*Df)) < 2 (for multilinearity:
vif(linearModeFinalForCheckingVif)
# No multicolinearity

originalDatasetToSave <- physicalAndEnvironmentalDataOriginal[, factorsFromRegression]
write.csv(originalDatasetToSave, "ProcessedVariablesToFit.csv")



# Test log linear model=========
selectedVariableDf <- rTreeFilteredVariables[, factorsFromRegression]
selectedVariableDf <- selectedVariableDf %>%
  transform(SalePrice = log(SalePrice + abs(min(SalePrice)) + 1))

logLinModel <- lm(SalePrice ~ ., data = selectedVariableDf)
summary(logLinModel)
plot(logLinModel)
# Not bad fit
# Multiple R-squared:  0.9103,	Adjusted R-squared:  0.9072 




# Attempt at a basic model


