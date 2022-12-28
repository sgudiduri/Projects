import pandas as pd
import numpy as np

class Data():
    STOCKS_PATH = './UI/Data/StocksPrices_wSentiment_PricePred_WithSentiment.csv'
    CLUSTERS_PATH = './UI/ClusteredTopicsSummary_Reduced.csv'
    
    def __init__(self):
      self.stocks = None
      self.tickers = None
      self.clusters = None
      self.cluster_tickers = None
      self.predictions = None
      self.performance = None

    
    def load_data(self, randomPrediction=False, randomCorrelations=False):
        stocks = pd.read_csv(self.STOCKS_PATH)
        pred_WOS = pd.read_csv('./UI/Data/StocksPrices_wSentiment_PricePred_WithoutSentiment.csv')     

        
        stocks['Date'] = pd.to_datetime(stocks['Date'])
        stocks = stocks.rename( columns={'Prediction_Last_40Days':'predicted_withSentiment'})
        stocks['predicted_withoutSentiment'] = pred_WOS['Prediction_Last_40Days']   

        tickers = stocks['Stock'].unique()
        
        self.stocks, self.tickers =  stocks, tickers
        
        self.performance = self.best_performer()

    def best_performer(self):
        performance = []

        cur_stock = None
        countWithoutSentiment = 0
        countWithSentiment = 0
        totalWithSentiment = 0 
        totalWithoutSentiment = 0
        last_price = 0
        count = 0


        for row in self.stocks.itertuples(index=True, name='ds'):
            if not row.predicted_withSentiment > 0: #excluding NaN
                continue
            
            count += 1
            totalWithSentiment += abs(row.Close - row.predicted_withSentiment)
            totalWithoutSentiment += abs(row.Close - row.predicted_withoutSentiment)
            
            if cur_stock is None or cur_stock != row.Stock:
                
                if cur_stock is not None: # Not first iteration
                    performance.append( [cur_stock, countWithSentiment, countWithoutSentiment, totalWithSentiment, totalWithoutSentiment, count])
                    
                last_price = 0
                cur_stock = row.Stock                
                countWithoutSentiment = 0
                countWithSentiment = 0
                totalWithSentiment = 0 
                totalWithoutSentiment = 0        
                count = 0
                continue
            
            if (last_price > 0 ):
                
                if ((last_price > row.Close and last_price > row.predicted_withoutSentiment) or 
                    (last_price < row.Close and last_price < row.predicted_withoutSentiment)):
                    countWithoutSentiment += 1
                    
                if ((last_price > row.Close and last_price > row.predicted_withSentiment) or 
                (last_price < row.Close and last_price < row.predicted_withSentiment)):
                    countWithSentiment += 1
            
            last_price = row.Close
            
            
        performance.append( [cur_stock, countWithSentiment, countWithoutSentiment, totalWithSentiment, totalWithoutSentiment, count])          
        
        return pd.DataFrame(performance, columns = ['Stock', 'RightMovementWithSentiment', 'RightMovementcountWithoutSentiment', 'TotalErrorWithSentiment', 'TotalErrorWithoutSentiment', 'Count'])
        
        
    def save(self):
        tk = pd.DataFrame(self.tickers, columns = ['Ticker'])
        tk.to_csv('./UI/Data/tickers.csv')
        self.performance.to_csv('./UI/Data/performance.csv')
        self.stocks.to_csv('./UI/Data/Stocks.csv')
        
        
if __name__ == '__main__':
    print('Loading data and processing ...')
    dt = Data()
    dt.load_data(randomPrediction=True)
    
    print( 'Saving data...') 
    dt.save()
        