# importing libraries and their associated
from sklearn.datasets import load_digits

digits = load_digits()
print(digits)
# Visualizing data
# Determining the total number of images and labels
print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

# Assigning the figure size
plt.figure(figsize=(20, 4))
# loop for showing images for 5 data(i.e. 0 to 4 number image).
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

# Splitting data set into training data set and test data set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Creating the model
from sklearn.linear_model import LogisticRegression

LogisticRegr = LogisticRegression(solver='liblinear', multi_class='ovr')

# Training the model
print(LogisticRegr.fit(X_train, Y_train))

print(LogisticRegr.predict(X_test[0].reshape(1, -1)))

print(LogisticRegr.predict(X_test[0:10]))

# Testing the model
predictions = LogisticRegr.predict(X_test)

# Finding the accuracy of the prediction
score = LogisticRegr.score(X_test, Y_test)
print(score)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Building the confusion matrix
cm = metrics.confusion_matrix(Y_test, predictions)
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted value')
all_sample_title = 'Accuracy score : {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

# Printing some of the test data images
index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, Y_test):
    if predict == actual:
        misclassifiedIndex.append(index)
    index += 1
plt.figure(figsize=(20, 3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
    plt.subplot(1, 4, plotIndex + 1)
    plt.imshow(np.reshape(X_test[wrong], (8, 8)), cmap=plt.cm.gray)
    plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], Y_test[wrong]), fontsize=15)
plt.show()

print(X_train.shape)

print(X_test.shape)
