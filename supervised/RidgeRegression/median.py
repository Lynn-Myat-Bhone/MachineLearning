import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RidgeRegression:
    def __init__(self,alpha=1.0):
        self.theta = None
        self.alpha = alpha
    
    def fit(self,x,y):
        X = np.c_[np.ones((x.shape[0], 1)), x]
        
        # using closed-form solution 
        identity_matrix = np.eye(X.shape[1])
        self.theta = np.linalg.inv(X.T @ X + self.alpha * identity_matrix) @( X.T @y)
         
    def predict(self, X):
        # Add a bias term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Make predictions
        return X_bias @ self.theta
          

data = pd.read_csv("https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/RidgeRegression/housing.csv")

x = data[['median_income']]
y = data['median_house_value']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#standardize is important in Ridge
scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train)
x_test_scal = scaler.fit_transform(x_test)

# Create and fit the Ridge Regression model
ridge = RidgeRegression(alpha=1.0)
ridge.fit(x_train_scal, y_train)

# Predictions
y_pred_train = ridge.predict(x_train_scal)
y_pred_test = ridge.predict(x_test_scal)

# Visualize predictions vs actual values
plt.scatter(x_test.values, y_test, label='Actual Values')
plt.scatter(x_test.values, y_pred_test, color='red', label='Predicted Values')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Ridge Regression: Actual vs Predicted Values")
plt.legend()
plt.show()
