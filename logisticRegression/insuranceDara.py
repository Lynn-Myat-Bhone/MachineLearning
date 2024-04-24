import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#read data
df  = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv')

# defint x and y
x = df['age'].values.reshape(-1,1)
y = df['bought_insurance']

#split data
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=122)

#build model and train
model = LogisticRegression()
model.fit(x_train,y_train)


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


accuracy = model.score(x_test,y_test) # accuracy 
print(accuracy)

# probabilities
model.predict_proba(x_train)
model.predict_proba(x_test)