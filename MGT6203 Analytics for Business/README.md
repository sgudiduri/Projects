# Home Price Prediction in Ames, Iowa
## Team 84
## Overview
This is the final project for MGT6203 Summer 2022 Team 084. Home Price Prediction in Ames, Iowa

## Setup the environment
1. [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. Create an environment where the default python version is 3.10. There are couple of options for this. Do ONLY ONE of these:
    1. [Install Python 3.10](https://www.python.org/downloads/) and make it the default python version on your system.
    2. Create a Python 3.10 conda environment. 
4. Clone the Team-84 repository from [git](https://github.gatech.edu/MGT-6203-Summer-2022-Canvas/Team-84)
5. [Install R](https://cran.r-project.org/bin/) or [Install RStudio] (https://www.rstudio.com/products/rstudio/download/) 

## Download the data from Kaggle
The data is already downloaded in the notebook folder in the repository. Follow the steps if a new copy of the data is required from kaggle:
1. Open a terminal or command prompt and navigate to Notebooks folder of repository (i.e. AmesHousing/Notebooks folder)
2. Run the following command: python -m notebook
3. This opens a python notebook environment. Open the DownloadFromKaggle.ipynb
4. Follow the instructions in the notebook to create a kaggle account and setup a kaggle api token
5. Run all the steps "Run All"
6. This should download train.csv and test.csv in the Notebooks folder
Note: The KaggleApi download might encounter error if the kaggle token in not setup properly

## Scrape data from www.neighborhoodscout.com
1. Open a terminal or command prompt and navigate to Scripts folder of repository (i.e. AmesHousing/Scripts folder)
2. Run the python script ScrapingWebPage.py
3. This should download Ames Neighborhood Rankings.csv in the Scripts folder

## Combine AmesHousing dataset and Ames Neighborhood Ranking.csv
1. Open a terminal or command prompt and navigate to Scripts folder of repository (i.e. AmesHousing/Scripts folder)
2. Run the following command: python -m notebook
3. This opens a python notebook environment. Open the Joining Environmental Datasets To Housing.ipynb
4. Run all the steps "Run All"
5. This should create PhysicalAndEnvironmental.csv in the Scripts folder

## Exploratory data analysis 
The team conducted a data analysis to see the data distribution, outliers, etc
1. Open a terminal or command prompt and navigate to Scripts folder of repository (i.e. AmesHousing/Scripts folder)
2. Run the following command: python -m notebook
3. This opens a python notebook environment. Open the ExploratoryAnalysis.ipynb
4. Run all the steps "Run All"

## Approach 1 - Using Python with All features
1. Open a terminal or command prompt and navigate to Notebooks folder of repository (i.e. AmesHousing/Notebooks folder)
2. Run the following command: python -m notebook
3. This opens a python notebook environment. Open the Implementation7.ipynb
4. Run all the steps "Run All"

## Approach 2 - Using R with feature selection
1. Open R or RStudio 
2. Open the AmesHousing/Scripts/Combined DataSet Data Analysis No Transformation and Better Scaling.R
3. Run the script
