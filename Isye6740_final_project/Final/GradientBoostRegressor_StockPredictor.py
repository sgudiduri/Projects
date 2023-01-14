
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, explained_variance_score

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
        
        print(f"Train Accuracy: {r2_score(y_train, train_predictions)}, Test Accuracy: {r2_score(y_test, test_predictions)}")
        print(f"Train Expected Variance: {explained_variance_score(y_train, train_predictions, multioutput='uniform_average')},Test Expected Variance: {explained_variance_score(y_test, test_predictions, multioutput='uniform_average')}")
        print(f"Train MSE: {mean_squared_error(y_train, train_predictions, multioutput='uniform_average')}, Test MSE: {mean_squared_error(y_test, test_predictions, multioutput='uniform_average')}")