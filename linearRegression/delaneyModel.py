import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#read data
df  = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

#define x and y
y = df['logS']
x = df.drop('logS',axis=1) 

#split data to train and test
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.2,random_state=122)

#Linear Regression
model = LinearRegression()
model.fit(x_train,y_train)

#prediction
y_train_pred = model.predict(x_train)

# evaluate model performance
# for training set
model_train_mse = mean_squared_error(y_train,y_train_pred)
model_train_r2 = r2_score(y_train,y_train_pred)

table = pd.DataFrame(["LinearRegression", model_train_mse, model_train_r2]).transpose()
table.columns=["Method","MSE(Train)","R2(Train)"]
print(table)

# Plot the CSV data and the linear regression line
line = np.polyfit(y_train,y_train_pred,1)
p = np.poly1d(line)
plt.scatter (y_train,y_train_pred,c='green',alpha = 0.5)# alpha ajust transparent of data point
plt.plot(y_train, p(y_train), color = 'red')  # Plotting the regression line
plt.ylabel('Predict logS')
plt.xlabel('logS')
plt.show()
