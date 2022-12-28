#%% Import Libs
import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
#import seaborn as sns
import yfinance as yf
import numpy as np
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
#import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
#import matplotlib.dates as mdates # Formatting dates
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
import joblib
import os
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,OneHotEncoder,MinMaxScaler


os.chdir('''Team_094\\''')

#%%
data = pd.read_csv('SentimentModelScripts/Outputs/Visualizations/Data/StocksPrices_wSentiment.csv')
#stockname = "GME"

def build_model(df,stockname,seqlen):
    os.chdir('''C:\\Users\\ayman\\Documents\\GitHub\\GT\DVA\\Team_094\\''')
    df.Date = pd.to_datetime(df['Date'])

    df = df[df['Stock'] == stockname]
    df_org = df.copy()
    
    del df['Predicted_Cluster_Sentiment']
    del df['Predicted_Reference_Sentiment']
    del df['Adj Close']
    del df['Stock']
    
    df = df.set_index("Date")
    target_column = ['Close'] 
    predictors = list(set(list(df.columns))-set(target_column))
    
    train_df = df.sort_values(by=['Date']).copy()
    date_index = train_df.index
    d = pd.to_datetime(df.index)
    df['Month'] = d.strftime("%m").astype("int")
    df['Year'] = d.strftime("%Y").astype("int") 
    train_df = train_df.reset_index(drop=True).copy()
    
    data_filtered = pd.DataFrame(train_df)

    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['Close']
    
    # Get the number of rows in the data
    nrows = data_filtered.shape[0]
    
    # Convert the data to numpy values
    np_data_unscaled = np.array(data_filtered)
    np_data = np.reshape(np_data_unscaled, (nrows, -1))
    print(np_data.shape)
    
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)
    
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = MinMaxScaler()
    df_Close = pd.DataFrame(data_filtered_ext['Close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    

    data = data = pd.DataFrame(df)
    
    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = seqlen
    
    # Prediction Index
    index_Close = data.columns.get_loc("Close")
    
    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
    
    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]
    
    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
        
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y
    
    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)
    
    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    # Validate that the prediction value and the input match up
    # The last close price of the second input sample should equal the first prediction value
    print(x_train[1][sequence_length-1][index_Close])
    print(y_train[0])
    

    # Configure the neural network model
    model = Sequential()
    
    # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
    n_neurons = x_train.shape[1] * x_train.shape[2]
    print(n_neurons, x_train.shape[1], x_train.shape[2])
    model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) 
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(5))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Training the model
    epochs = 20
    batch_size = 20
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stop]
                       )
#    os.chdir('..')
    os.chdir('''StockModelScripts\\FinalModel\\Model\\WSentiment\\''')          
    model_name = stockname+'_LSTM_Model'
    model.save(model_name)
    
    # Get the predicted values
    y_pred_scaled = model.predict(x_test)
    
    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
        
    # Mean Absolute Error (MAE)
    MAE = mean_absolute_error(y_test_unscaled, y_pred)
    print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')
    
    # Mean Absolute Percentage Error (MAPE)
    MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
    print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')
    
    # Median Absolute Percentage Error (MDAPE)
    MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
    print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')
    
    df_org['index'] = np.arange(len(df_org))
    
    
    prediction_set = pd.DataFrame(y_pred).reset_index()

    prediction_set['index'] = prediction_set['index'] + (len(df_org)-len(prediction_set))
    
    df_org = df_org.merge(prediction_set,on = ['index'], how='left')
    
    del df_org['index']
    df_org = df_org.rename(columns={0: 'Prediction_Last_40Days'})
    
    return df_org



#%%
os.chdir('''SentimentModelScripts\\''')
df_referencetickers = pd.read_csv('Dependencies/StockTickers.csv')

df_completestock_pred=pd.DataFrame()


for i in df_referencetickers['Stock_Search_List']:
    print(i)
    try:
        df_completestock_pred_prev = build_model(df = data, stockname = i, seqlen=50)
        df_completestock_pred = df_completestock_pred.append(df_completestock_pred_prev,ignore_index=True)
    except Exception as e:
        print(e)
        print('retry:',i)
        try:
            df_completestock_pred_prev = build_model(df = data, stockname = i, seqlen=25)
            df_completestock_pred = df_completestock_pred.append(df_completestock_pred_prev,ignore_index=True)
        except Exception as e:
            print(e)
            pass
        pass
    

#%%
os.chdir('''..''')
os.chdir('''..''')
os.chdir('''..''')
os.chdir('''..''')
os.chdir('''SentimentModelScripts\\''')
df_completestock_pred.to_csv('Outputs/Visualizations/Data/StocksPrices_wSentiment_PricePred_WithSentiment.csv',index=False)
