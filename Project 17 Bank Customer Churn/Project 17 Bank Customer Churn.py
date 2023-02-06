'''
This code is building a machine learning model to predict customer churn in a bank using a 
dataset stored in the "Bank Customer Churn Prediction.csv" file.  It uses a Random Forest 
classifier algorithm from the scikit-learn library. The code performs several steps to 
prepare the data for modeling listed below.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Bank Customer Churn Prediction.csv')
print("Bank Customer Churn Prediction.csv dataset:")
print(dataset)
x = dataset.iloc[:, 1:11].values 	# [all data - customer ID - Churn]
y = dataset.iloc[:, 11].values		# Churn (customer left = 0, customer stayed = 1)

# preparing array data to be standard scaled
def sc_arr_prep(arr):
	for x_idx in range(len(arr)):
		for y_idx in range(len(arr[0])):
			data = arr[x_idx][y_idx]
			try:
				arr[x_idx][y_idx] = float(data)
			except:
				data_temp = 0
				for idx in range(len(data)):
					data_temp = float(data_temp + ord(data[idx]))
				arr[x_idx][y_idx] = data_temp

sc_arr_prep(x)


# Setup data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 50, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
T_x_test = sc.transform(x_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
T_x_test = pca.transform(T_x_test)
explained_variance = pca.explained_variance_ratio_

# Fitting the SVC classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
RandomForestClass = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
model = RandomForestClass.fit(x_train, y_train)


# Predicting the Test set results
y_pred = model.predict(T_x_test)
print("Predicting (float) x_test set:\n", x_test)
print("Prediction:\n", y_pred)
print("Results:\n", y_test)

def accuracy(arr_test, arr_train):
	correct_count = 0
	total_count = len(arr_test)
	for idx in range(len(arr_test)):
		if (arr_test[idx] == arr_train[idx]):
			correct_count += 1
	print("accuracy =", correct_count, "/", total_count, "=", (correct_count/total_count), "\n")

accuracy(y_pred, y_test)

# Predict specific case (used in training)
test_arr = [[619, 'France', 	'Female', 	42, 2, 0.0, 		1, 1, 1, 101348.88],	# = 1
			[608, 'Spain', 		'Female', 	41, 1, 83807.86, 	1, 0, 1, 112542.58], 	# = 0
			[502, 'France', 	'Female', 	42, 8, 159660.8, 	3, 1, 0, 113931.57],	# = 1
			[699, 'France', 	'Female', 	39, 1, 0.0, 		2, 0, 0, 93826.63], 	# = 0
			[850, 'Spain', 		'Female', 	43, 2, 125510.82, 	1, 1, 1, 79084.1], 		# = 0
			[645, 'Spain', 		'Male', 	44, 8, 113755.78, 	2, 1, 0, 149756.71],	# = 1
			[822, 'France', 	'Male', 	50, 7, 0.0, 		2, 1, 1, 10062.8], 		# = 0
			[376, 'Germany', 	'Female', 	29, 4, 115046.74, 	4, 1, 0, 119346.88],	# = 1
			[501, 'France', 	'Male', 	44, 4, 142051.07, 	2, 0, 1, 74940.5], 		# = 0
			[684, 'France', 	'Male', 	27, 2, 134603.88, 	1, 1, 1, 71725.73]] 	# = 0
print("Predicting test_arr set:")
for idx in range(len(test_arr)):
	print(test_arr[idx])
sc_arr_prep(test_arr)
test_arr = sc.transform(test_arr)
test_arr = pca.transform(test_arr)
test_pred = model.predict(test_arr)
train_arr = dataset.iloc[0:10, 11].values
print("Prediction:\n", test_pred)
print("Results:\n", train_arr)

accuracy(test_pred, train_arr)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
def Visualizer(x_set, y_set, title):
	X_set, y_set = x_train, y_train
	X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
	                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
	plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
	             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
	plt.xlim(X1.min(), X1.max())
	plt.ylim(X2.min(), X2.max())
	for i, j in enumerate(np.unique(y_set)):
	    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
	                c = ListedColormap(('red', 'green'))(i), label = j)
	plt.title(title)
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.legend()
	plt.get_current_fig_manager().window.state('zoomed')
	plt.show()
title = 'Random Forest Classifier (Training set)'
Visualizer(x_train, y_train, title)
title = 'Random Forest Classifier (Test set)'
Visualizer(x_test, y_test, title)