'''
This code performs regression analysis on a big data set using a linear regression model 
from the scikit-learn library. The goal is to first predict the dependent variable (PE) based 
on the independent variables (AT, V, AP, RH) in the dataset. The code splits the data 
into training and testing sets, trains the model on the training set, predicts the 
dependent variable using the test set, and finally calculates the R^2 score, which is a measure 
of the model's accuracy. The R^2 score is found to be 0.93, indicating a strong fit 
between the model and the data.
'''
# Project 6 R^2 Big Data Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import dataset
dataset = pd.read_csv('Data.csv')
print("Data.csv dataset:")
print(dataset)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [-1]].values	#shape (-1,1)

# Setup data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Training the Lienar regression model on the whole dataset
Linear_regressor = LinearRegression()
Linear_model = Linear_regressor.fit(x_train, y_train)

# Predicting y test
y_linear_pred = Linear_model.predict(x_test)

# R^2 score using test values
Linear_score = r2_score(y_test, y_linear_pred)
print("Linear_R^2_score =", Linear_score)