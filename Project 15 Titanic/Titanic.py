'''
This code performs a logistic regression machine learning model to 
predict the survival of passengers from the Titanic disaster, based 
on various features such as passenger ID, gender, class, fare, etc.
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Visualising the Test set results
from matplotlib.colors import ListedColormap
def Visualizer(Title, SS_x, x, y):
	y_pred = Log_model.predict(SS_x)
	X_set = []
	for x_idx in range (len(x)):
		X_set.append(x[x_idx][0])

	plt.title(Title)
	plt.xlabel('Passenger ID')
	plt.ylabel('Survived')
	plt.scatter(X_set, y_pred)
	plt.get_current_fig_manager().window.state('zoomed')
	plt.show()

# Importing the dataset
Survived_dataset = pd.read_csv('gender_submission.csv')
Passenger_data = pd.read_csv('test.csv')
print("gender_submission.csv dataset:")
print(Survived_dataset)
print("test.csv dataset:")
print(Passenger_data)

# We dont care about the names
Passenger_data.drop(['Name'], axis = 1, inplace = True)

Passenger_data = np.array(Passenger_data)

x = []

for x_idx in range(len(Passenger_data)):
	row = []
	for y_idx in range(len(Passenger_data[0])):

		if (Passenger_data[x_idx][y_idx] == 'male'):
			row.append(0)
		elif (Passenger_data[x_idx][y_idx] == 'female'):
			row.append(1)

		elif (y_idx == 8):	#cabin
			try:
				row.append(ord((Passenger_data[x_idx][y_idx])[0]))
			except:
				row.append(-1)

		elif (y_idx == 9):	# Embarked
			row.append(ord(Passenger_data[x_idx][y_idx]))

		elif type(Passenger_data[x_idx][y_idx]) is str:
			try:
				row.append(int(Passenger_data[x_idx][y_idx]))
			except:
				row.append(-1)

		elif (Passenger_data[x_idx][y_idx] >= 0):
			row.append(Passenger_data[x_idx][y_idx])

		else:
			row.append(-1)
	x.append(row)

y = Survived_dataset.iloc[:, 1].values 	# Survived	(0 or 1)


#Setup data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Feature scalling
from sklearn.preprocessing import StandardScaler
SS_x_train = StandardScaler().fit_transform(x_train)
SS_x_test = StandardScaler().fit_transform(x_test)

#Training the LogisticRegression model on the whole dataset
from sklearn.linear_model import LogisticRegression
LogisticRegressor = LogisticRegression(random_state = 0, solver='lbfgs')
Log_model = LogisticRegressor.fit(SS_x_train, y_train)

# Predicting survivalbility of test set
y_log_pred = Log_model.predict(SS_x_test)
print("Predicting survivalbility of test set (1 alive, 0 death):")
print(y_log_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_log_pred)
print(cm)

Train_Title = 'LogisticRegressor (Training set)'
Test_Title = 'LogisticRegressor (Test set)'

Visualizer(Train_Title, SS_x_train, x_train, y_train)
Visualizer(Test_Title, SS_x_test, x_test, y_test)