import numpy as np
import pandas as pd

from sklearn import svm

import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(font_scale=1.2)

recipes = pd.read_csv('Cupcakes vs Muffins.csv')
print(recipes.head())

# plot our data
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.show()

# if type=Muffin then type_label will be 0 else 1
type_label = np.where(recipes['Type'] == "Muffin", 0, 1)

# As we don't want the data for type which is in 0th column of the dataset.
# so we take column from to end using recipes.columns.values[1:]
# tolist() is used to show the values in list
recipe_features = recipes.columns.values[1:].tolist()

# To get all the values of all columns in recipe_features we use the following:
# ingredients = recipes[recipe_features].values

# To get all the values of any column(ex: Flour and Sugar) we use the following:
ingredients = recipes[['Flour', 'Sugar']].values
print(ingredients)

# fit model
model = svm.SVC(kernel='linear')
model.fit(ingredients, type_label)

# Get separating hyperplane
w= model.coef_[0]
a= -w[0]/ w[1]
xx= np.linspace(30, 60)
# y=mx+c is done as below:
yy= a* xx - (model.intercept_[0])/w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b= model.support_vectors_[0]
yy_down = a*xx +(b[1] - a*b[0])
b= model.support_vectors_[-1]
yy_up= a*xx +(b[1] - a *b[0])

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx,yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
# k-- is used for dotted line
plt.plot(xx, yy_up, 'k--')
plt.show()

# create a function to predict muffin or cupcake
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print("You are looking at a muffin recipe!")
    else:
        print("You are looking at a cupcake recipe!")

muffin_or_cupcake(50,20)

# Plot this on the graph
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50,20, 'yo', markersize='9')
plt.show()