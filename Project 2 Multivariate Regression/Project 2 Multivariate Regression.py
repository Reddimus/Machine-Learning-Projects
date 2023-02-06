'''
The code is implementing a multiple linear regression model for predicting the profit of a startup based on the 
R&D Spend, Administration and Marketing Spend. The input data is read from a .csv file and the categorical data 
in the State column is encoded using OneHotEncoding. The data is then split into a training set and a test set. 
The model is trained using the training set and then tested by making predictions on the test set. The predictions 
are compared to the actual values. Finally, the relationship between R&D Spend and profit is visualized for both 
the training set and the test set.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('50_Startups.csv')
print("50_Startups.csv:\n", dataset)
Categ_data = dataset.iloc[:, :-1].values
# X will grab all columns except the last column (col 0-3)
Profit = dataset.iloc[:, -1].values
# y will grab the last column which is named profit 

# (OneHotEncoding) Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder		#(col 0 -> 2 = 3 cols)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
Categ_data = np.array(ct.fit_transform(Categ_data))

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
Categ_data_train, Categ_data_test, Profit_train, Profit_test = train_test_split(Categ_data, Profit, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Categ_data_train, Profit_train)

# predicting the test set results
y_pred = regressor.predict(Categ_data_test)
np.set_printoptions(precision=2)
print("Predicting the test set results:")
print(np.concatenate((y_pred.reshape(len(y_pred),1), Profit_test.reshape(len(Profit_test),1)),1))
'''
# Multivariable graph in 2D:
R_N_D_train = Categ_data_train[:, [3]]
R_N_D_test = Categ_data_test[:, [3]]

# Visualising the Training set results
plt.scatter(R_N_D_train, Profit_train, color = 'red')
plt.plot(R_N_D_train, regressor.predict(Categ_data_train), color = 'blue')
plt.title(' vs  (Training set)')
plt.xlabel('')
plt.ylabel('')
plt.show()

# Visualising the Test set results
plt.scatter(R_N_D_test, Profit_test, color = 'red')
plt.plot(R_N_D_test, regressor.predict(Categ_data_test), color = 'blue')
plt.title(' vs (Test set)')
plt.xlabel('')
plt.ylabel('')
plt.show()
'''