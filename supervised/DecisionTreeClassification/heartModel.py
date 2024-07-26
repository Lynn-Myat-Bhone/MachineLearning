import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/siddiquiamir/Data/master/heart.csv")

# df.isnull()
# df.isnull().sum()

X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

x_train,y_train,x_test,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(criterion="gini",max_depth=7,min_samples_split=9,random_state=10)

model.fit(x_train,y_train) 

y_pred = model.predict(x_test)

accuracy_score(y_test,y_pred)