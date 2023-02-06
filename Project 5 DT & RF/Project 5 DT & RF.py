'''
This code is a simple example of using the Random Forest Regression 
algorithm to predict a salary based on the level of a position.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
print("Position_Salaries.csv dataset:")
print(dataset)
x = dataset.iloc[:, [1]].values 	#level shape(-1, 1)
y = dataset.iloc[:, 2].values		#salary

#Training the Decision Tree Regression model on the whole dataset
RanForestRegressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
DTRF_model = RanForestRegressor.fit(x, y)

#Predicting a new result (level 6.5)
y_RF_pred = DTRF_model.predict([[6.5]])
print("Predicting level 6.5 pay = $", int(y_RF_pred))

#Visualising the DTRF_model results
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, DTRF_model.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()