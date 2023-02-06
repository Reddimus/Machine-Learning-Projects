'''
This code performs a binary classification task to predict if a customer 
would subscribe to a term deposit based on their age and job.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('bank-direct-marketing-campaigns.csv')
print("bank-direct-marketing-campaigns.csv dataset:")
print(dataset)
x = dataset.iloc[:, [0, 1]].values 	# [Age, job]
y = dataset.iloc[:, -1].values		# Subscribed


# assign a key to job(s)
for x_idx in range(len(x)):
    job = x[x_idx][1]
    x[x_idx][1] = ord(job[0]) + ord(job[1]) + ord(job[-1])


# assign 1/0 to yes/no
for idx in range(len(y)):
    if (y[idx] == 'yes'):
        y[idx] = int(1)
    else:
        y[idx] = int(0)

y = y.astype('int')

# Setup data (decreased training data & test data to run faster; increase data for better accuracy)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 1/50, test_size = 1/50, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

# Fitting the SVC classifier to the Training set
from sklearn.svm import SVC
SVC_class = SVC(kernel = 'rbf', random_state= 0)
SVC_model = SVC_class.fit(x_train, y_train)

# predicting the test set results
y_SVC_pred = SVC_model.predict(x_test)
print("Predicting the test set results:")
print(y_SVC_pred)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
SVC_cm = confusion_matrix(y_test, y_SVC_pred)
print("Confusion matrix:")
print(SVC_cm)

#Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVC_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVC (Training set)')
plt.xlabel('Age')
plt.ylabel('job (key)')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()

#Visualising the Training test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, SVC_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.4, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVC (Test set)')
plt.xlabel('Age')
plt.ylabel('job (key)')
plt.legend()
plt.get_current_fig_manager().window.state('zoomed')
plt.show()