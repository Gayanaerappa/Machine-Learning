import numpy as np
from sklearn.cluster import KMeans

# Input data: [Age, Income]
X = np.array([
    [25, 30000],
    [30, 40000],
    [35, 50000],
    [45, 80000],
    [50, 90000],
    [23, 28000]
])

# Create K-Means model (2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0)

# Fit the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

print("Cluster labels:", labels)
