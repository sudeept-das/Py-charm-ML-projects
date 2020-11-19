# Loading the library with the iris dataset
from sklearn.datasets import load_iris

# Loading scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Loading pandas
import pandas as pd

# Loading numpy
import numpy as np

# Setting random seed
np.random.seed(0)

# Creating an object called iris with the iris data
iris = load_iris()
print(iris)
# Creating a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# Viewing the top 5 rows
print(df.head())

# Adding a new column for the species name
df['species'] = pd.Categorical.from_codes(iris.target,
                                          iris.target_names)
# Viewing the top 5 rows
print(df.head())

# Creating Test and Train Data

df['is_train'] = np.random.uniform(0,1,len(df)) <= .75
# View the top 5 rows
print(df.head())

# Creating dataframes with test rows and training rows
train, test = df[df['is_train']==True] , df[df['is_train']==False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))

# Create a list of the feature column's names
features = df.columns[:4]
print(features)

# Converting each species name into digits
y= pd.factorize(train['species'])[0]
# Viewing target
print(y)

# Creating a random forest Classifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
#Training the classifier
clf.fit(train[features], y)
print(clf)

# Applying the trained classifier to the test
print(clf.predict(test[features]))

print(test[features])
print(features)

# Viewing the predicted probabilities for the first 20 observations
print(clf.predict_proba(test[features])[0:20])

# mapping names for the plants for each predited plant class
preds = iris.target_names[clf.predict(test[features])]

# View the PREDICTED species for the first 25 observations
print(preds[0:25])

# Creating the Confusion matrix
print(pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']))

preds = iris.target_names[clf.predict([[5.0,3.6,1.4,2.0],[5.0,1.6,1.4,5.0]])]
print(preds)