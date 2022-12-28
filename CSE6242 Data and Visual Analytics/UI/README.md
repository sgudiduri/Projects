# Preparing data for visualization

1. Copy the CSV files from `SentimentModelScripts/Outputs/Visualizations/Data` to `UI\Data`
2. Run data_processing.py

The data processing will 
1. Include the prediction data with and without sentiment in the stock price table
2. Extract unique ocurrences of stock
3. Analyse the prediction performance

## Data exported

What will be exported:

`tickers.csv` - Unique Stocks mentioned in `Stocks`
`Stocks.csv` - Containing stocks prices and predictions
`performance.csv` - Dataset containing the summary of predictions performance.

