# importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')
print(len(dataset))
print(dataset.head())
# Replace zeroes
zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','Insulin']
for column in zero_not_accepted:
    # Replacing NAN by 0
    dataset[column] = dataset[column].replace(0, np.NaN)

    mean = int(dataset[column].mean(skipna=True))
    # Replacing NAN by mean of that column
    dataset[column] = dataset[column].replace(np.NaN, mean)

# Split dataset into training data and testing data
X= dataset.iloc[:,0:8]          # X stores the datas of all rows which are in column 0 to 7
y= dataset.iloc[:,8]            # Y stores the datas of all rows which are in 8
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0, test_size=0.2)

# Feature Scaling
sc_X = StandardScaler()
# We fit the scalar with X_train set and also transform the X_test set
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

# Define the model:init KNN
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')

# Fit Model
print(classifier.fit(X_train, y_train))

# Predicting the test results
y_pred = classifier.predict(X_test)
print(y_pred)

#Evaluate Model
cm= confusion_matrix(y_test, y_pred)
print(cm)

print(f1_score(y_test,y_pred))

print(accuracy_score(y_test,y_pred))
