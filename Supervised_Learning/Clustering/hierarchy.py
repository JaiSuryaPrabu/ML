import pandas as pd

# Importing the dataset
print("Importing the dataset")
dataset = pd.read_csv('customers.csv')
print(dataset.head())
X = dataset.iloc[:, [3, 4]].values
print("Splitting into independent values")

print("Importing the hierarchy class")
from sklearn.cluster import AgglomerativeClustering
# Training the K-Means model on the dataset
heirachy = AgglomerativeClustering(n_clusters = 5)
y_heir = heirachy.fit_predict(X)
print("Model is trained")

from matplotlib import pyplot as plt
plt.scatter(X[y_heir == 0, 0], X[y_heir == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_heir == 1, 0], X[y_heir == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_heir == 2, 0], X[y_heir == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_heir == 3, 0], X[y_heir == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_heir == 4, 0], X[y_heir == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(heirachy.cluster_centers_[:, 0], heirachy.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()