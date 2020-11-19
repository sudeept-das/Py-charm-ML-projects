# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset and Extrating the Independent and Dependent variables
companies= pd.read_csv('1000_Companies.csv')
X = companies.iloc[:, :-1].values
Y = companies.iloc[:, 4].values

companies.head()

# Data Visualisation
# Building the Correlation matrix
sns.heatmap(companies.corr())
plt.show()

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3]= labelencoder.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()
print(X)

# Avoiding Dummy variable trap
# Removing 1st column
X=X[:,1:]

# Splitting the data into Train and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression Model to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(n_jobs=1)
print(regressor)
regressor.fit(x_train, y_train)

# Predicticting the test set results
y_pred = regressor.predict(x_test)
print(y_pred)

#Calculating the coefficients and Intercept
print(regressor.coef_)
print(regressor.intercept_)

# Calculating the R squared value
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
