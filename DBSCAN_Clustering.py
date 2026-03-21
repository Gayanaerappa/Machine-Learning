# Unsupervised Learning - DBSCAN Clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Step 1: Dataset (Age, Spending Score)
X = np.array([
    [22, 20], [25, 30], [47, 70], [52, 65],
    [46, 60], [56, 65], [55, 60], [60, 70],
    [18, 15], [20, 18], [90, 5]  # outlier
])

# Step 2: Create DBSCAN model
model = DBSCAN(eps=10, min_samples=2)

# Step 3: Train model
labels = model.fit_predict(X)

# Step 4: Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma')

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering (with Outliers)")

plt.show()

# Step 5: Output
print("Cluster Labels:", labels)

# Show noise points (-1)
noise_points = X[labels == -1]
print("Noise Points:", noise_points)
