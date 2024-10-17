import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/refs/heads/main/Supervised/SupportVectorMachines/housing.csv')
print(f'Dataset size: {df.shape[0]} rows')
print(df.head())

# Define features and target
X = df[['median_income']].values
y = df['median_house_value'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit model using SVR with a linear kernel
model = SVR(kernel='linear')  # Removed random_state
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
