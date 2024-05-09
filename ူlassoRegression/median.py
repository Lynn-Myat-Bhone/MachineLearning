import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

data = pd.read_csv("https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/RidgeRegression/housing.csv")

x = data[['median_income']]
y = data['median_house_value']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train_scal  = x_train.s
model = Lasso(alpha=1)
model.fit(x_train,y_train)
predict = model(x_trian_scal)