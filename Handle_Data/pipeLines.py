#this model doesn't need to preprocess cuz dataset already suitable for model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_iris()
X, y = data.data, data.target
#if you need to preprocess data , here is example code-
    # # Example data columns
    # numerical_cols = ['Age', 'Salary']
    # categorical_cols = ['Gender', 'City']

    # # Define transformers for each type of data
    # numerical_transformer = StandardScaler()  # For numerical data
    # categorical_transformer = OneHotEncoder()  # For categorical data

    # # Combine transformers into a ColumnTransformer
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numerical_transformer, numerical_cols),  # Apply StandardScaler to numerical columns
    #         ('cat', categorical_transformer, categorical_cols)  # Apply OneHotEncoder to categorical columns
    #     ])

    # # Define the model
    # model = RandomForestClassifier()

    # # Create a pipeline with preprocessing and modeling
    # pipeline = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('model', model)
    # ])

    # Fit the pipeline on training data
    # pipeline.fit(X_train, y_train)


# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier())
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict new data
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = pipeline.predict(new_data)
print(f"Predicted class: {prediction}")
