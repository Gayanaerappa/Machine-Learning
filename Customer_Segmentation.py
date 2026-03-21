# Unsupervised Machine Learning - KMeans Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Dataset (no labels)
# Example: Age vs Spending Score
X = np.array([
    [22, 20], [25, 30], [47, 70], [52, 65],
    [46, 60], [56, 65], [55, 60], [60, 70],
    [18, 15], [20, 18]
])

# Step 2: Create model
kmeans = KMeans(n_clusters=2, random_state=0)

# Step 3: Train model
kmeans.fit(X)

# Step 4: Get clusters
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 5: Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()

# Step 6: Output clusters
print("Cluster Labels:", labels)
print("Centroids:", centroids)
