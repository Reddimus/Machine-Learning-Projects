'''
This is a code that uses the Support Vector Regression (SVR) machine learning 
algorithm to predict salary based on the position level in an organization.
'''
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
print("Position_Salaries.csv dataset:")
print(dataset)
x = dataset.iloc[:, [1]].values 	#level shape(-1, 1)
y = dataset.iloc[:, 2].values		#salary

#Feature Scaling
StandardScaled_x = StandardScaler().fit_transform(x.reshape(-1, 1))
StandardScaled_y = StandardScaler().fit_transform(y.reshape(-1, 1))


#Training the SVR model on the whole dataset
SVR_rbf_regressor = SVR(kernel = 'rbf')
'''
rbf is set as default
Radial basis function (RBF) is a function 
whose value depends on the distance 
(usually Euclidean distance) to a center 
(xc) in the input space. The most commonly 
used RBF is Gaussian RBF. It has the same 
form as the kernel of the Gaussian probability 
density function
'''
SVR_model = SVR_rbf_regressor.fit(StandardScaled_x, StandardScaled_y.ravel())
y_SVR_SS_pred = SVR_model.predict(StandardScaled_x)
print("Prediction at multiple levels:")
print(y_SVR_SS_pred)

#Predicting a new result (level 6.5)
SS_6_5 = (StandardScaled_x[5] + StandardScaled_x[6]) / 2
y_SVR_pred_6_5 = SVR_model.predict([SS_6_5])
print("Prediction at level 6.5")
print(y_SVR_pred_6_5)

#Visualising the Standard scaled SVR results
plt.scatter(StandardScaled_x, StandardScaled_y, color = 'red')
plt.plot(StandardScaled_x, y_SVR_SS_pred, color = 'blue')
plt.title('Regression Results')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
