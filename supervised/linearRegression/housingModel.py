import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegressioin:
    def __init__(self,learning_rate=0.01,print_interval=None, num_iterations = 1000):
        
        """
        Parameters :
            Learning_rate : float, optional (default=0.01)
                             the step size in updating the model parameter
            
            print_interval : int or none,optional(default=None)
                             If provided, print the training process (loss,parameters) every `print_interval` iteartion.
                             Set to None to disable printing during training
            
            num_iterations:  int, optional (default=1000)
                             The number of iterations specifies how many times the gradient descent loop will be executed.
                             More iterations may lead to a more accurate model, but there's a trade-off with computational cost.
        """
        self.learning_rate = learning_rate
        self.print_interval = print_interval
        self.num_iteration = num_iterations
        self.weight = None
        self.bias = None
    
    def fit(self,x,y):
        """
            fit the model to training data
        """
        x = np.array(x)
        y = np.array(y).flatten()
        self.weight = np.zeros(x.shape[1])
        self.bias = 0
        
        for iteration in range(self.num_iteration):
            y_pred = np.dot(x, self.weight) + self.bias
            error = y_pred - y
            
             # Update weights and bias using gradient descent
            self.weight -= self.learning_rate * (1 / x.shape[0]) * np.dot(x.T, error)
            self.bias -= self.learning_rate * (1 / x.shape[0]) * np.sum(error)
            
            #Print training progress
            if self.print_interval and iteration % self.print_interval==0:
                loss = self.calculate_loss(x,y)
                print(f"Itreation {iteration}, Loss:{loss}")
                print(f"Weight : {self.weight}")
                print(f"Bias : {self.bias}")
    
    def predict(self,x):
        """
        Make predictions using the trained linear regression model.
        """
        x = np.array(x)
        y_pred = np.dot(x,self.weight) + self.bias
        return y_pred.reshape(-1,1)
    
    def calculate_loss (self,x,y):
        """
        Calculate the mean squared error loss for the current model.
        """
        y_pred = np.dot(x, self.weight) + self.bias
        errors = y_pred - y
        loss = np.mean(errors**2)
        return loss
    
#read data 
csv_file = 'https://raw.githubusercontent.com/Mattral/ML-AI-Algorithms-from-scratch/main/Supervised/LinearRegression/housing.csv'
data = pd.read_csv(csv_file)
x_csv = data[['median_income']].values # input values
y_csv = data[['median_house_value']].values # target variables

# build model
model = LinearRegressioin(learning_rate=0.01,num_iterations=1000,print_interval=100)
model.fit(x_csv,y_csv)

# Make predictions using CSV data
predictions_csv = model.predict(x_csv)

# Plot the CSV data and the linear regression line
plt.scatter(x_csv, y_csv, label='CSV Data')  # Scatter plot of the original data
plt.plot(x_csv, predictions_csv, color='green', label='Linear Regression')  # Plot the linear regression line
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear Regression with CSV Data')
plt.legend()
plt.show()