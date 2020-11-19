# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset and Extrating the Independent and Dependent variables
companies= pd.read_csv('1000_Companies')
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