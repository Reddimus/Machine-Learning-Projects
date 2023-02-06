'''
This code performs a simple linear regression analysis to model the relationship between 
a person's years of experience and their salary. The data set is loaded from a .csv file, 
split into training and testing sets, and used to train a linear regression model. The 
model is then used to predict the salary based on years of experience for the test data 
set. Finally, the relationship between years of experience and salary is visualized for 
both the training and test sets using scatter plots.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# import data set and translate to data frame
Data_set = pd.read_csv('Salary_Data.csv')
print("Data_set:")
print(Data_set)

# translate columns/ sections into lists/ vectors of data
X = Data_set.iloc[:, [1]].values

y = Data_set.iloc[:, 0].values

# Setup data
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
# Training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Predicting the Test set results:")
print(y_pred)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience', color = 'white')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()