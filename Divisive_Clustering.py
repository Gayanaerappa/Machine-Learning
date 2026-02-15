import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [8, 8],
    [9, 9],
    [10, 8]
])

# First split
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("First Division")
plt.show()

# Further split cluster 0
cluster0 = X[labels == 0]

kmeans2 = KMeans(n_clusters=2, random_state=0)
labels2 = kmeans2.fit_predict(cluster0)

plt.scatter(cluster0[:, 0], cluster0[:, 1], c=labels2)
plt.title("Second Division (Cluster 0)")
plt.show()
