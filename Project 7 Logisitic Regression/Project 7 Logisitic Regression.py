'''
This code trains and evaluates a logistic regression model on a dataset of social network 
ads to predict whether a customer will purchase the product or not based on their age and 
estimated salary. The code imports the required libraries, loads the data from a CSV file, 
splits the data into training and testing sets, and applies feature scaling. The logistic 
regression model is then trained on the training set and its predictions are compared to 
the actual values in the test set using a confusion matrix. Finally, the code visualizes 
the results of the logistic regression model in the form of a contour plot for both the 
training and test sets.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print("Social_Network_Ads.csv dataset:")
print(dataset)
x = dataset.iloc[:, [0, 1]].values 	# [Age, EstimatedSalary]
y = dataset.iloc[:, 2].values		# Purchased

# Setup data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature scalling
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

#Training the LogisticRegression model on the whole dataset
LogisticRegressor = LogisticRegression(random_state = 0, solver='lbfgs')
Log_model = LogisticRegressor.fit(x_train, y_train)
y_log_pred = Log_model.predict(x_test)

# Predict a new specific result (age: random, Salary: random)
'''
Age = random.randrange(16, 60)
EstimatedSalary = random.randrange(15000, 200000) 
Age = 36
EstimatedSalary = 200000
#SS_Age_EstSal_arr = StandardScaler().fit_transform([[Age, EstimatedSalary]])
y_log_pred_21_50k = StandardScaler().inverse_transform(Log_model.predict([[Age, EstimatedSalary]]))
print("With an age of", Age, "and the estimated Salary of", EstimatedSalary, "our Logistic Regression model predicts:")
if y_log_pred_21_50k == 1:
	print(y_log_pred_21_50k, "(Will purchase the product)")
elif y_log_pred_21_50k == 0:
	print(y_log_pred_21_50k, "(Will NOT purchase the product)")
'''

#Confusion Matrix
cm = confusion_matrix(y_test, y_log_pred)
print("Confusion Matrix:")
print(cm)

#Visualising the Logistic Regression Training set results
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, LogisticRegressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualising the Logistic Regression Training test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, LogisticRegressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()