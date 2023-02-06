'''
The code is a script for regression analysis, specifically Polynomial Regression 
and Linear Regression, on an employee salary dataset. It uses libraries such as 
numpy, pandas, matplotlib, and scikit-learn. The code trains the Linear Regression 
and Polynomial Regression models on the dataset and makes predictions. The 
prediction results are visualized with matplotlib.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Training the Polynomial Regression model on the whole dataset
def Poly_Train (resolution):
	poly = PolynomialFeatures(degree = resolution, include_bias = False)
	poly_model = poly.fit_transform(X.reshape(-1, 1))
	model = LinearRegression().fit(poly_model,y)
	y_poly_pred = model.predict(poly_model)
	return y_poly_pred


# Visualising the Regression results
def visualise(y_pred):
	if (y_pred[9] > 750000):
		Regression_type = "Polynomial Regression"
	else:
		Regression_type = "Linear Regression"
	plt.figure(figsize=(10, 6))
	plt.title(Regression_type, size=16)
	plt.scatter(X, y)
	plt.plot(X, y_pred)
	plt.show()


# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)
X = dataset.iloc[:, [1]].values 	#level
y = dataset.iloc[:, -1].values	#salary

#Training the Linear Regression model on the whole dataset
regressor = LinearRegression()
regressor.fit(X, y)
y_linear_pred = regressor.predict(X)

#Training the Polynomial Regression model on the whole dataset
y_poly_pred = Poly_Train(4)

# Visualising the Linear Regression results
visualise(y_linear_pred)

# Visualising the Polynomial Regression results
visualise(y_poly_pred)

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
y_poly_pred = Poly_Train(8)
visualise(y_poly_pred)

# Predicting a new result with Linear Regression
Linear_model = LinearRegression().fit(X, y)
Linear_pred_6_5 = Linear_model.predict(np.array(6.5).reshape(-1, 1))
print("At level 6.5 our Linear Regression model predicts a salary of $", Linear_pred_6_5[0])


# Predicting a new result with Polynomial Regression
poly = PolynomialFeatures(degree = 8, include_bias = False)
poly_model = poly.fit_transform(X.reshape(-1, 1))
model = LinearRegression().fit(poly_model,y)

lv_six_Poly_model = poly.fit_transform(np.array(6.5).reshape(-1, 1))
Poly_pred_6_5 = model.predict(lv_six_Poly_model)
print("At level 6.5 our Polynomial Regression model predicts a salary of $", Poly_pred_6_5[0])