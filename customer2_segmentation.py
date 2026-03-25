# Customer Segmentation using K-Means Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create dataset (Annual Income, Spending Score)
X = np.array([
    [15, 39], [16, 81], [17, 6], [18, 77], [19, 40],
    [20, 76], [21, 6], [22, 94], [23, 3], [24, 72],
    [25, 14], [26, 82], [27, 32], [28, 61], [29, 31]
])

# Step 2: Apply KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Step 3: Get labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 4: Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")
plt.show()
