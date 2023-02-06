'''
This code applies the Decision Tree and Random Forest classifiers to a social 
network ads dataset to predict whether a user has purchased a product or not.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print("Social_Network_Ads.csv dataset:")
print(dataset)
x = dataset.iloc[:, [0, 1]].values 	# [Age, EstimatedSalary]
y = dataset.iloc[:, 2].values		# Purchased

#setup data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

#Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClass = DecisionTreeClassifier(criterion = 'entropy', random_state =0)
DecisionTree_model = DecisionTreeClass.fit(x_train, y_train)

# Predicting the Test set results
y_DecisionTree_pred = DecisionTree_model.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("Decision Tree confusion matrix:")
DecisionTree_cm = confusion_matrix(y_test, y_DecisionTree_pred)
print(DecisionTree_cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, DecisionTree_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


# Visualising the Test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, DecisionTree_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


# Fitting the RandomForest classifier into the Training set
from sklearn.ensemble import RandomForestClassifier
RandomForestClass = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
RandomForest_model = RandomForestClass.fit(x_train, y_train)


# Predicting the RandomForest test set results
y_RandomForest_Pred = RandomForest_model.predict(x_test)


# Making the RandomForest Confusion Matrix 
from sklearn.metrics import confusion_matrix
print("Random Forest Confusion Matrix:")
RandomForest_cm = confusion_matrix(y_test, y_RandomForest_Pred)
print(RandomForest_cm)

# Visualising the Training set results
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, RandomForest_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()


# Visualising the Test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, RandomForest_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()