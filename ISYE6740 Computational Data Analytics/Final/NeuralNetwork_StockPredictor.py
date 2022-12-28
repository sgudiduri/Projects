import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense
from matplotlib.pyplot import figure
import keras.losses
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, explained_variance_score

class NeuralNetwork_StockPredictor:        
            
    def fit_predict(self, mos, X_train, y_train, X_test, y_test):
        
        # Standardize the train and test features
        scaler = MinMaxScaler()
        scaled_train_features = scaler.fit_transform(X_train)
        scaled_test_features = scaler.fit_transform(X_test)
      
        # Create loss function
        def sign_penalty(y_true, y_pred):
            penalty = 100.
            loss = tf.where(tf.less(y_true * y_pred, 0), \
                             penalty * tf.square(y_true - y_pred), \
                             tf.square(y_true - y_pred))

            return tf.reduce_mean(loss, axis=-1)
        
        # enable use of loss with keras
        keras.losses.sign_penalty = sign_penalty  
        
        print(keras.losses.sign_penalty)

        # Create the model
        model = Sequential()
        model.add(Dense(50, input_dim=scaled_train_features.shape[1], activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='linear'))

        # Fit the model
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(scaled_train_features, y_train, epochs=50, verbose=0)

        # Plot the losses from the fit
        plt.plot(history.history['loss'])

        # Use the last loss as the title
        plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
        plt.show()
        
        # Calculate R^2 score
        train_predictions = model.predict(scaled_train_features)
        test_predictions = model.predict(scaled_test_features)
        
        print(f"Train Accuracy: {r2_score(y_train, train_predictions)}, Test Accuracy: {r2_score(y_test, test_predictions)}")
        print(f"Train Expected Variance: {explained_variance_score(y_train, train_predictions, multioutput='uniform_average')},Test Expected Variance: {explained_variance_score(y_test, test_predictions, multioutput='uniform_average')}")
        print(f"Train MSE: {mean_squared_error(y_train, train_predictions, multioutput='uniform_average')}, Test MSE: {mean_squared_error(y_test, test_predictions, multioutput='uniform_average')}")

        # Plot predictions vs actual
        plt.scatter(train_predictions, y_train, label='train')
        plt.scatter(test_predictions, y_test, label='test')
        plt.legend()
        plt.show()
        