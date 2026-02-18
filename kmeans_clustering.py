import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -----------------------------
# 1. Sample Dataset (No Labels!)
# -----------------------------

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [8, 8],
    [9, 9],
    [10, 8]
])

print("Dataset:")
print(X)

# -----------------------------
# 2. Train K-Means Model
# -----------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# -----------------------------
# 3. Get Cluster Labels
# -----------------------------

labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("\nCluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# -----------------------------
# 4. Visualization
# -----------------------------

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(centers[:, 0], centers[:, 1], marker='X')
plt.title("K-Means Clustering (Unsupervised Learning)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
