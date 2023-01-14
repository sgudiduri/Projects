"""
=====================================
Data Loader
=====================================

This class is responsible for creating
data for stock market optimizer and stock marker
price predictor
"""

from os import path
from pathlib import Path 
import pandas as pd
from urllib.request import Request, urlopen
import bs4 as bs
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
        
'''
    Utility function to remove irrelevent scraped columns. 
'''    
   
def reformat_ticker_dataframe(ticker_history):
    cols = {i:i.replace("_close","") for i in list(ticker_history.columns) if "_close" in i}
    ticker_history = ticker_history[[i for i in list(cols.keys())]]
    ticker_history = ticker_history.rename(columns=cols)
    return ticker_history        

    
'''
    get_sp_index method is responsible for retrieve 
    all the stocks avaiable in market. This is done via web scraping. 
    Delete any file in resources/utility/data/{}.xlsx location to fetch
    new data
'''
def get_sp_index():
    try:            
        sp_index_path =  Path('data/sp_index.xlsx') 
        if(sp_index_path.is_file()):
            df = pd.read_excel('data/sp_index.xlsx')
            df = df[["Company","Symbol","Weight","Price", "Chg"]]
        else:
            req = Request('https://www.slickcharts.com/sp500', headers={'User-Agent': 'Mozilla/5.0'})
            webpage = urlopen(req).read()
            soup = bs.BeautifulSoup(webpage,'lxml')
            table = soup.find('table', attrs={'class':'table table-hover table-borderless table-sm'})
            df = pd.read_html(str(table))[0]
            df = df[["Company","Symbol","Weight","Price","Chg"]]
            df.to_excel("data/sp_index.xlsx")
        return df
    except:       
        return "Hello World"


'''
    calls Yahoo api to retriever all stock market historical
    data. For this project we are only including S&P index tickers. 
    Delete any file in resources/utility/data/{}.xlsx location to fetch
    new data        
'''    
def get_ticker_historical(symbol_list):
    try:
        ticker_history_file = Path('data/ticker_history.xlsx')  
        if ticker_history_file.is_file():
            ticker_history = pd.read_excel('data/ticker_history.xlsx')
            ticker_history = ticker_history.set_index("Date")
            return reformat_ticker_dataframe(ticker_history.dropna())
        else:                
            ticker_history = pd.DataFrame(list(), columns=[])   
            for i in symbol_list:
                ticker_df = yf.download(i, start="2017-03-30", end="2022-03-31")[["Close"]]    
                if len(ticker_df) > 500:        
                    ticker_df = ticker_df.rename(columns={"Close": f"{i}_close"})
                    ticker_history = ticker_df.join(ticker_history) 
                    
            ticker_history.to_excel("data/ticker_history.xlsx")
            return reformat_ticker_dataframe(ticker_history.dropna())
    except:       
        return None
 
'''
    This method is responsible to retieving data for selected tickets. 
'''
def get_selected_ticker_historical(symbol_list):
    top_3_stocks_dict = {}
    for stock in symbol_list:
        top_3_stocks_dict[stock] = yf.download(stock, start="2015-03-30", end="2022-03-31")[["Close", "Volume"]]
        
    return top_3_stocks_dict

'''
    This method is responsible to creating training and feature dataset. 
'''
def create_features_training(mos):
    # Create 10-day % changes of Adj_Close for the current day, and 5 days in the future
    mos['10d_future_close'] = mos['Close'].shift(-5)
    mos['10d_close_future_pct'] = mos['10d_future_close'].pct_change(5)
    mos['10d_close_pct'] = mos['Close'].pct_change(5)

    import talib
    feature_names = ['10d_close_pct']  # a list of the feature names for later

    # Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
    for n in [14,30,50,180]:

        # Create the moving average indicator and divide by Adj_Close
        mos['ma' + str(n)] = talib.SMA(mos['Close'].values,
                                  timeperiod=n) / mos['Close']
        # Create the RSI indicator
        mos['rsi' + str(n)] = talib.RSI(mos['Close'].values, timeperiod=n)

        # Add rsi and moving average to the feature name list
        feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

    print(feature_names)

    # Create 2 new volume features, 1-day % change and 5-day SMA of the % change
    new_features = ['Volume_10d_change', 'Volume_10d_change_SMA']
    feature_names.extend(new_features)
    mos['Adj_Volume_10d_change'] = mos['Volume'].pct_change()
    mos['Adj_Volume_10d_change_SMA'] = talib.SMA(mos['Adj_Volume_10d_change'].values,
                            timeperiod=10)
    
    mos= mos.drop(columns=["Volume"])
    mos= mos.dropna()
    return mos
    
'''
    Graphs Monthly prive data      
'''    
def visualize_historical_ticker(df, ticker_list):
    try:
        plt.figure(figsize=(10,8))
        for i in ticker_list:
            plt.plot(df[i])
        plt.legend(df.columns,fontsize=16)
        plt.xlabel("Months",fontsize=18)
        plt.ylabel("Stock price (Monthly average)",fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.show()
    except:       
        return None