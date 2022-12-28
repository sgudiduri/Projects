
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

class GradientBoostRegressor_StockPredictor:    
    
            
    def fit_predict(self, mos, X_train, y_train, X_test, y_test):
      
        # Create GB model -- hyperparameters have already been searched for you
        gbr = GradientBoostingRegressor(max_features=4,
                                        learning_rate=0.01,
                                        n_estimators=200,
                                        subsample=0.6,
                                        random_state=42)
        gbr.fit(X_train, y_train)
        print(gbr.score(X_train, y_train))
        print(gbr.score(X_test, y_test))

        # Make predictions with our model
        train_predictions = gbr.predict(X_train)
        test_predictions = gbr.predict(X_test)

        # Create a scatter plot with train and test actual vs predictions
        plt.scatter(y_train, train_predictions, label='train')
        plt.scatter(y_test, test_predictions, label='test')
        plt.legend()
        plt.show()
        
        print(f"Train Accuracy Score: {gbr.score(X_train, y_train)}")
        print(f"Test Accuracy Score: {gbr.score(X_test, y_test)}")