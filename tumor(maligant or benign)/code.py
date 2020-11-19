import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('tumor.csv')
print(data.head())

sns.jointplot('radius_mean', 'texture_mean', data=data)
# shows graph and histogram of texture_mean vs radius_mean for the dataset data
plt.show()

# heat shows graph for comparing all the features of the dataset using color for the dataset data with each other
sns.heatmap(data.corr())
plt.show()

# Checking for null datas
print(data.isnull().sum())

# X contains all worst and y contains only diagnosis
X = data.values[:,22:]
y = data.values[:,2]


from sklearn.model_selection import train_test_split

# We have taken test_size=0.3 to take 30% of the data(X,y) in testing data(X_test, y_test) and 70% in the training
# data(X_train, y_train) that we are going to split
X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=101, test_size=0.3)
print("X_test length:",len(X_test))
print("X_train length:",len(X_train))
print("y_test length:",len(y_test))
print("y_train length:",len(y_train))

# Creating the Model
from sklearn.linear_model import LogisticRegression
LogModel = LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=1)
print(X_train.shape)
print(y_train.shape)
print(LogModel.fit(X_train, y_train))

# Predicting for the model
y_pred = LogModel.predict(X_test)
print(y_pred)

# Checking the classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
