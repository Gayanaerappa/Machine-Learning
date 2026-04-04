import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Step 1: Dataset
data = {
    'X': [1, 2, 3, 8, 9, 10, 25, 26, 27],
    'Y': [2, 3, 4, 9, 10, 11, 30, 31, 32]
}

df = pd.DataFrame(data)

# Step 2: Dendrogram
linked = linkage(df, method='ward')

plt.figure()
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Step 3: Apply Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=3)
df['Cluster'] = model.fit_predict(df)

# Step 4: Output
print(df)

# Step 5: Visualization
plt.scatter(df['X'], df['Y'], c=df['Cluster'])
plt.title("Hierarchical Clustering")
plt.show()
