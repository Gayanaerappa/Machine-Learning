# Unsupervised Learning - Hierarchical Clustering

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Step 1: Dataset (Age, Spending Score)
X = np.array([
    [22, 20], [25, 30], [47, 70], [52, 65],
    [46, 60], [56, 65], [55, 60], [60, 70],
    [18, 15], [20, 18]
])

# Step 2: Create dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(6, 4))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# Step 3: Apply Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X)

# Step 4: Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering")

plt.show()

# Step 5: Output
print("Cluster Labels:", labels)
