import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create DMatrix for the training data
churn_dmatrix = xgb.DMatrix(X_train, label=y_train)

# Define parameters for XGBoost
params = {
    'objective': 'multi:softmax',   # multi-class classification
    'num_class': 3,                 # number of classes in the target variable
    'eval_metric': 'mlogloss'       # evaluation metric (log loss for multi-class)
}

# Perform cross-validation
cv_results = xgb.cv(
    dtrain=churn_dmatrix,           # DMatrix containing training data and labels
    params=params,                  # Dictionary of parameters for XGBoost
    nfold=3,                        # Number of cross-validation folds
    num_boost_round=5,              # Number of boosting rounds (trees)
    metrics="mlogloss",             # Metric to evaluate in each round; "mlogloss" for multi-class
    as_pandas=True,                 # Return results as a pandas DataFrame
    seed=42                        # Seed for reproducibility
)

# Print cross-validation results
print("Cross-validation results:")
print(cv_results)

# Calculate the final test accuracy from the last boosting round's error mean
test_error_mean = cv_results["test-mlogloss-mean"].iloc[-1]  # Mean log loss from the last round

# Train the final model using the optimal number of boosting rounds
final_model = xgb.train(params, churn_dmatrix, num_boost_round=len(cv_results))

# Create a DMatrix for the test set
dtest = xgb.DMatrix(X_test)

# Make predictions on the test set
y_pred = final_model.predict(dtest)

# Evaluate and print the test accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Accuracy: {accuracy:.2%}")
