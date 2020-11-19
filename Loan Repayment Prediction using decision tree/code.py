# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Loading data file
balance_data = pd.read_csv(
    'C:/Users/User/PycharmProjects/Loan Repayment Prediction using decision tree/Decision_Tree_ Dataset.csv', sep=",",
    header=0)

print("Dataset Length:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)
print("Dataset:: ")
print(balance_data.head())

# Seperating target variables
X = balance_data.values[:, 0:4]
Y = balance_data.values[:, 5]
print(X)
print(Y)
# Splitting Dataset into Test and Train
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, random_state=100)

#Function to perform training with Entropy
clf_entropy= DecisionTreeClassifier(criterion= "entropy", random_state= 100,
                                    max_depth=3, min_samples_leaf=5)
print(clf_entropy.fit(X_train,y_train))

#Function to make predictions
y_pred_en = clf_entropy.predict(X_test)
print(y_pred_en)

#Checking Accuracy
print("Accuracy is: ", accuracy_score(y_test,y_pred_en)*100)
