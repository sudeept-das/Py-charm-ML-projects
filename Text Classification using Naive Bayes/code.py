import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
print(data.target_names)

# Defining all the categories
categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']
# Training the data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories
test = fetch_20newsgroups(subset='test', categories=categories)

# Printing training data, testing data and length of the data
print(train.data[5])
print(test.data[5])
print(len(train.data))

# importing necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer  # Use to tokenize the words of the dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Creating a model based on Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model with the train data
model.fit(train.data, train.target)

# Creating labels for the test data
labels = model.predict(test.data)

# Creating confusion matrix and heat map
from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
            yticklabels=train.target_names)

# Plotting Heatmap of Confusion Matrix
plt.xlabel('true label')
plt.ylabel('predict label')
plt.show()


# predicting category on new data based on trained model
def predict_category(s, train=train, model=model):
    # s is a string which we need to predict and return the target_names(i,e. category)
    pred = model.predict([s])
    return train.target_names[pred[0]]


print(predict_category('Jesus Christ'))
print(predict_category('Sending load to International Space Station'))
print(predict_category('BMW is better than Audi'))
print(predict_category('President of India'))



