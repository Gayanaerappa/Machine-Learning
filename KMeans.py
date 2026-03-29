import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Load dataset
df = pd.read_csv("/mnt/data/Electric_Vehicle_Population_Data (1).csv")

# View first few rows
print(df.head())
features = df[['Model Year', 'Electric Range', 'Base MSRP']]

# Handle missing values
features = features.fillna(features.mean())
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to dataset
df['Cluster'] = clusters
print(df.groupby('Cluster')[['Model Year', 'Electric Range', 'Base MSRP']].mean())
plt.scatter(
    scaled_features[:, 0],
    scaled_features[:, 1],
    c=clusters,
    cmap='viridis'
)
plt.xlabel('Model Year (scaled)')
plt.ylabel('Electric Range (scaled)')
plt.title('K-Means Clustering of EV Data')
plt.colorbar(label='Cluster')
plt.show()
