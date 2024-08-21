import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
 
df = pd.read_csv('https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/Naive%20Bayes/spam_ham_dataset.csv')

# Separate features (X) and labels (y)
X = df['text']  # Replace 'email_text' with the correct column name ('text' in this case)
y = df['label_num']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
 
# Train the model
gnb.fit(X_train, y_train)
 
# Predict the labels for the test set
y_pred = gnb.predict(X_test)
 
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')