import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/DecisionTrees/housing.csv")

print(data['median_income'].describe())
x = data['median_income'].values.reshape(-1,1)
y = data['median_house_value'].values
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print('Pediction',prediction)
print('actual',y_test)

#evaluate model
mse = mean_squared_error(y_test, prediction)
print(f"Mean Squared Error: {mse}")
