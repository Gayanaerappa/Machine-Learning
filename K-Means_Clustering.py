# K-Means Clustering with Elbow Method

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Dataset (Age, Spending Score)
X = np.array([
    [22, 20], [25, 30], [47, 70], [52, 65],
    [46, 60], [56, 65], [55, 60], [60, 70],
    [18, 15], [20, 18]
])

# Step 2: Elbow Method to find best K
wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 3: Apply K-Means (choose K=2 from elbow)
model = KMeans(n_clusters=2, random_state=0)
labels = model.fit_predict(X)
centroids = model.cluster_centers_

# Step 4: Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()

# Step 5: Output
print("Cluster Labels:", labels)
print("Centroids:\n", centroids)
