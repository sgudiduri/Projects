# Team_094


### Background

In this project our team decided to run an experiment to understand the effect social media conversations and their underlying sentiment (specifically on the Wall Street Bets subreddit) may have had on the price of stocks.

### Usage

Navigate to UI directory ex - `cd Team_094/UI/`
Open the Tableau file `StocksPricePredictionSentiment.twbx`

 Tableau Reader may be needed in order to view the final visualization UI. The link to this free software may be found here https://www.tableau.com/products/reader/download
 
 Should the user encounter any issues while opening the package, an alternative location for the end visualization may be found here (albeit with limited functionality)
 https://public.tableau.com/app/profile/ayman.saeed/viz/StocksPricePredictionSentiment/StockSentimentDashboard?publish=yes

### Steps to recreate:

This repo will help you recreate the work we did and further contribute to the project.

#### Install dependencies:
  1. Navigate to SentimentModelScripts/Dependencies
  2. Run the following statement to install your relevant python libraries to run the project: pip install -r requirements.txt
 


#### Sentiment Model scripts:

##### Background:
    Navigate to SentimentModelScripts directory ex - cd Team_094/SentimentModelScripts

    These scripts will help you:
      1. Ingest data from the WallStreetBets subreddit.
      2. Clean that data and prepare it for sentiment modelling. Additionally apply an external model (Cardiff NLP's Twitter Sentiment model) to get an initial base line for sentiment classification.
      3. Cluster the comments under key themes found in the corpus and visualize them.
      4. Create clusters/partitions based on direct references to certain stocks in each text.
      5. Split data into randomized sample for human labelling and rejoin it when completed.
      6. Train and build a specialized RoBERTa model finetuned to WallStreetBets (WSB) data. Also retrain the model as needed.
      7. Run a data pipeline script that uses the models you built to run inferences against data and consolidate the data to be further used downstream on stock market predictions.

###### 1. Ingest data from the WallStreetBets subreddit:
          1. Navigate to - Data/scrape.py
          2. Run the script to output your corpus to WSB_Comments.csv (if you want to run delta please modify the dates and output csv to WSB_Comments_Delta.csv) ex - python Data/scrape.py
          3. You are now ready to clean your data for downstream tasks

###### 2. Clean data:
          1. Navigate to - Models/Scripts
          2. Run the script "1 - dataengineering.py" to clean your data for enhanced sentiment and topic modelling: ex- python Models/Scripts/1 - dataengineering.py
          3. The output of this process will be stored at this location for further downstream tasks: Team_094/SentimentModelScripts/Outputs/WSB_Comments_Clean.csv

###### 3. Cluster data and topic modelling:
          1. Navigate to - Models/Scripts
          2. Run the script "2 - clustermodel.py" to cluster your data. ex - python Models/Scripts/2 - clustermodel.py
          3. The outputs of this process will be stored at these locations (with their appropriate explanations) for further downstream tasks:

              Initial Clustering Model (for data exploration):
              - Models/Model/InitialClusteringModel (initial clustering model)
              - Outputs/WSB_Comments_Clean_wClusters.csv (data enhanced with initial clustering model - data can be used for data exploration)
              - Outputs/ExploratoryData/ClusteredTopicsSummary.csv (a summary of the initial clustering model's extracted topics including representational comments - data can be used for data exploration)
              - Outputs/ExploratoryData/ClusteredTopicsOverTime.csv (daily trending words per topic over the life of the corpus - data can be used for data exploration)
              - Outputs/Visualizations/clusters.html (all clusters visualized with their inter-topic distance - visualization can be used to explore key themes found in the text and their relationship to each other)
              - Outputs/Visualizations/clusterhierachy.html (all clusters visualized with their topic hierarchy - visualization can be used to explore key themes found in the text and their relationship to each other)

              Final Clustering Model (for final inference and visualization):
              - Models/Model/ClusteringModel (final clustering model that reduces the number of topics to a more manageable number for human interpretation and visualization)
              - Outputs/WSB_Comments_Clean_wClusters_Reduced.csv (data enhanced with the final clustering model - data can be used on downstream tasks)
              - Outputs/ExploratoryData/ClusteredTopicsSummary_Reduced.csv (a summary of the final clustering model's extracted topics including representational comments - data can be used for data exploration)
              - Outputs/Visualizations/Data/ClusteredTopicsSummary.csv (a summary of the final clustering model's extracted topics including representational comments - data can be used for the final visualization)
              - Outputs/Visualizations/clusters_r.html (final model's reduced clusters visualized with their inter-topic distance - visualization can be used to explore key themes found in the text and their relationship to each other)
              - Outputs/Visualizations/clusterhierachy_r.html (final model's reduced clusters visualized with their topic hierarchy - visualization can be used to explore key themes found in the text and their relationship to each other)
              - Outputs/ExploratoryData/ClusteredTopicsOverTime_Reduced.csv (final model's reduced clusters visualized with their topic hierarchy - visualization can be used to explore key themes found in the text and their relationship to each other)
              - Outputs/Visualizations/Data/TopicsOverTime.csv (daily trending words per final model's topics over the life of the corpus - data can be used for the final visualization)

          4. Run the script "2.5 - VisualizeClusters.py" to output clustered mapping for visualization. The outputs of this script can be found below:
             - Outputs/ExploratoryData/ClustersMapping_R.csv (for exploration)
             - Outputs/Visualizations/Data/ClustersMapping.csv (to be used in the final visualization)

###### 4. Cluster data and partition the data based on direct references:
          1. Navigate to - Models/Scripts
          2. Run the script "3 - clusterbysearch.py" to tag columns with a list of common stocks
          3. The output of this process will be stored at this location for further downstream tasks:
            - Outputs/ExploratoryData/ClustersMapping_R.csv (for exploration)
            - Outputs/Visualizations/Data/ClustersMapping.csv (to be used in the final visualization)

###### 5. Split the data for human labelling and rejoin it when labelling is completed:
          1. Navigate to - Sample/
          2. Run the script "1 - data_splitter.py" to partition and randomly sample data
          3. The output of this process will be stored in the Raw/ location for manual (human) labelling:
          4. After manual labelling is completed - run "2 - data_rejoinier.py" to rejoin the data and prepare it for sentiment modelling: Sample/Labelled/data.csv

###### 6. Train and build a specialized RoBERTa model finetuned to WSB data:
          1. The team used a Google Collab notebook to train the model due to lack of local gpus.

            - You may upload the following script "4 - roberta_wteamsentiment.py" and the following data "Sample/Labelled/data.csv" to run the script and train the model. The model should download the following zip file that you can unzip to store your model "sentiment_model_teamsentiment.zip". Unzip the model and store move everything under the unzipped files directory "content/" to the following directory "SentimentModelScripts\Models\Model\". You should now have a model folder "sentiment_model_teamsentiment" to use in your inferencing downstream.

            - However if your computer is able to run the model locally, run the following script "4 - roberta_wteamsentiment.py" to take the input above (Sample/Labelled/data.csv - you may need to change line 27 if running locally) to train the model. ex: python SentimentModelScripts\Models\Scripts\4 - roberta_wteamsentiment.py. Move the saved model into the directory mentioned above ("SentimentModelScripts\Models\Model\")

          2. If you want to train the existing model on additional data later run the following script "4.5 RobertaModel_Retrain.py" to update the model based on newly ingested data.

###### 7. Inference and Consolidate data to be fed downstream:
        1. Navigate to - Models/Scripts/
        2. Run the script "5 - Inference+Consolidation.py" to run all your models against the data and output a series of curated csvs to be used in visualzation downstream.
        3. The output of this process will be stored in the following locations:
          - Outputs/Visualizations/Data/StocksPrices_wSentiment.csv (Daily stock prices and mean weighted sentiment - audience * average comment sentiment per day - can be used downstream in visualization tasks)
          - Outputs/Visualizations/Data/Cluster_StocksComments.csv  (Comments per associated stock per cluster - can be used downstream in visualization tasks)
          - Outputs/Visualizations/Data/Reference_StocksComments.csv (Comments per associated stock per direct reference to that stock- can be used downstream in visualization tasks)

#### Stock Model scripts:
    Navigate to StockModelScripts directory ex - cd Team_094/StockModelScripts/FinalModel/Scripts
    These scripts will help you create models that predict stock prices with and without sentiment.

    1. Run the following script "Model_Inference_Consolidate_NoSentiment.py" to predict stock prices without sentiment.
      The output of this script will be a number of individual models under "FinalModel/Model/NoSentiment"
      The script will also output data to the following locations to be used in the downstream visualizations.

      - SentimentModelScripts/Outputs/Visualizations/Data/Outputs/Visualizations/Data/StocksPrices_wSentiment_PricePred_WithoutSentiment.csv

    2. Run the following script "Model_Inference_Consolidate_WithSentiment.py" to predict stock prices with sentiment.
      The output of this script will be a number of individual models under "FinalModel/Model/WSentiment"
      The script will also output data to the following locations to be used in the downstream visualizations.

      - SentimentModelScripts/Outputs/Visualizations/Data/Outputs/Visualizations/Data/StocksPrices_wSentiment_PricePred_WithSentiment.csv

#### UI Setup:
  Navigate to UI directory ex - cd Team_094/UI/
##### Preparing data for visualization

1. Copy the CSV files from `SentimentModelScripts/Outputs/Visualizations/Data` to `UI\Data`
2. Run data_processing.py

The data processing will
1. Include the prediction data with and without sentiment in the stock price table
2. Extract unique ocurrences of stock
3. Analyse the prediction performance

###### Data exported

What will be exported:

`tickers.csv` - Unique Stocks mentioned in `Stocks`
`Stocks.csv` - Containing stocks prices and predictions
`performance.csv` - Dataset containing the summary of predictions performance.




### Acknowledgments

In addition to the acknowledgments mentioned in the specific scripts. Some additional helpful guides that helped guide us through this project may be found below.
- BERT Transformer Models Transfer Learning for Multi-Class-Classification Guide
  - https://towardsdatascience.com/multi-class-classification-with-transformers-6cf7b59a033a
  - https://github.com/jamescalam/transformers/blob/main/course/project_build_tf_sentiment_model/01_input_pipeline.ipynb

- LSTM stock market
  - https://vannguyen-8073.medium.com/using-lstm-multivariate-and-univariate-in-order-to-predict-stock-market-39e8f6551c93
  - https://github.com/NourozR/Stock-Price-Prediction-LSTM/blob/master/StockPricePrediction.py
  - https://github.com/sibeltan/stock_price_prediction_LSTM/blob/master/Stock_Price_Prediction.ipynb
  - https://github.com/shimonyagrawal/Stock-Prices-Prediction-using-Keras-LSTM-Model/blob/main/Stock%20Price%20Prediction.ipynb
