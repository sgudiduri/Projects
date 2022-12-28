
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class RandomForestRegressor_StockPredictor:    
    
            
    def fit_predict(self, mos, X_train, y_train, X_test, y_test):
      
        rfr = RandomForestRegressor(n_estimators=200)
        rfr.fit(X_train, y_train)

        # Create a dictionary of hyperparameters to search
        grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4,8], 'random_state': [42]}
        test_scores = []

        # Loop through the parameter grid, set the hyperparameters, and save the scores
        for g in ParameterGrid(grid):
            rfr.set_params(**g)  # ** is "unpacking" the dictionary
            rfr.fit(X_train, y_train)
            test_scores.append(rfr.score(X_train, y_train))

        # Find best hyperparameters from the test score and print
        best_idx = np.argmax(test_scores)
        print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
        
        # Use the best hyperparameters from before to fit a random forest model
        rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=8, random_state=42)
        rfr.fit(X_train, y_train)

        # Make predictions with our model
        train_predictions = rfr.predict(X_train)
        test_predictions = rfr.predict(X_test)

        # Create a scatter plot with train and test actual vs predictions
        plt.scatter(y_train, train_predictions, label='train')
        plt.scatter(y_test, test_predictions, label='test')
        plt.legend()
        plt.show()
        
        print(f"Train Accuracy Score: {rfr.score(X_train, y_train)}")
        print(f"Test Accuracy Score: {rfr.score(X_test, y_test)}")
    
        