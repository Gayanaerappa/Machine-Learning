# -------------------------------
# DIVISIVE CLUSTERING (DIANA STYLE)
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# 1. LOAD DATA
file_path = "/mnt/data/Electric_Vehicle_Population_Data (1).csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# 2. SELECT NUMERICAL COLUMNS ONLY
num_df = df.select_dtypes(include=[np.number]).dropna()

print("Numerical Columns Used:")
print(num_df.columns)

# 3. STANDARDIZE DATA
scaler = StandardScaler()
X = scaler.fit_transform(num_df)

# 4. DISTANCE MATRIX
distance_matrix = squareform(pdist(X, metric='euclidean'))

# 5. DIVISIVE CLUSTERING FUNCTION
def divisive_clustering(dist_matrix, max_clusters=3):
    n = dist_matrix.shape[0]
    clusters = {0: list(range(n))}
    cluster_id = 1

    while len(clusters) < max_clusters:
        # Find cluster with maximum average distance
        split_cluster = max(
            clusters,
            key=lambda c: np.mean(dist_matrix[np.ix_(clusters[c], clusters[c])])
        )

        points = clusters[split_cluster]

        if len(points) <= 2:
            break

        # Find most distant point (splinter object)
        avg_dist = dist_matrix[points][:, points].mean(axis=1)
        splinter = points[np.argmax(avg_dist)]

        new_cluster = [splinter]
        remaining = [p for p in points if p != splinter]

        for p in remaining:
            if np.mean(dist_matrix[p, new_cluster]) < np.mean(dist_matrix[p, remaining]):
                new_cluster.append(p)

        clusters[split_cluster] = [p for p in points if p not in new_cluster]
        clusters[cluster_id] = new_cluster
        cluster_id += 1

    return clusters

# 6. RUN DIVISIVE CLUSTERING
clusters = divisive_clustering(distance_matrix, max_clusters=3)

# 7. ASSIGN CLUSTER LABELS
labels = np.zeros(X.shape[0])

for k, v in clusters.items():
    for idx in v:
        labels[idx] = k

df = df.loc[num_df.index]
df["Cluster"] = labels.astype(int)

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# 8. VISUALIZATION (FIRST TWO FEATURES)
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0], X[:, 1],
    c=labels,
    cmap='viridis',
    s=40
)
plt.xlabel(num_df.columns[0])
plt.ylabel(num_df.columns[1])
plt.title("Divisive Hierarchical Clustering")
plt.colorbar(label="Cluster")
plt.show()
