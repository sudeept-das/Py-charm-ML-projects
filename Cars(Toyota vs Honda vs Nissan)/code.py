# importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('cars.csv')

X = dataset[dataset.columns[:-1]]  # to take all rows but not tha last column of the dataset
print(X.info())


X = X.convert_objects(convert_numeric=True)  # To convert all data values of the dataset to numeric
print(X.info())

# Eliminating null values
for i in X.columns:
    X[i]=X[i].fillna(int(X[i].mean()))                   # Filling data inside any missing data with the mean of all the values of that column
print(X.info())

# Eliminating null values

for i in X.columns:
    print(X[i].isnull().sum())

# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(0, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the car dataset

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
X = X.as_matrix(columns=None)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Toyota')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Nissan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Honda')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')
plt.title('Clusters of car make')
plt.legend()
plt.show()
