import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
data = pd.read_csv('melb_data.csv')
y = data.Price
X = data.drop(['Price','Mount Waverley'],axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=42)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Approch 1
cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#drop columns
reduced_X_train = X_train.drop(cols_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))