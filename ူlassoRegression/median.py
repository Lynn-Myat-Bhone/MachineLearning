import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
        
data = pd.read_csv("https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/RidgeRegression/housing.csv")

x = data[['median_income']]
y = data['median_house_value']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train)
x_test_scal = scaler.fit_transform(x_test)

model = Lasso(alpha=1)
model.fit(x_train,y_train)
predict = model.predict(x_train_scal)

print(predict)

plt.scatter(x_test.values,y_test,label="actual values")
plt.scatter(x_train.values,predict,label="predicted values",color="red")
plt.show()