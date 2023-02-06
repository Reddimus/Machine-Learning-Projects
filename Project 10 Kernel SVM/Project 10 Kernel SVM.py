'''
This code performs a binary classification of a social network ads dataset using a 
Support Vector Machine (SVM) classifier. The SVM classifier is trained on two 
features: age and estimated salary of the users, and the target variable is whether 
the user has purchased a product or not (0 or 1).
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# SVC
from sklearn.svm import SVC

# Fitting the SVC classifier to the Training set
SVC_class = SVC(kernel = 'rbf', random_state= 0)
SVC_model = SVC_class.fit(x_train, y_train)

# predicting the test set results
y_SVC_pred = SVC_model.predict(x_test)
print("Predicting the test set results:")
print(y_SVC_pred)

# Confusion Matrix
print("Confusion Matrix")
SVC_cm = confusion_matrix(y_test, y_SVC_pred)
print(SVC_cm)


#Visualising the knn Training set results
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVC_class.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVC (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


#Visualising the knn Training test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVC_class.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVC (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()