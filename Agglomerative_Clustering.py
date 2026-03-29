# ===============================
# Agglomerative Clustering Code
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# 1. Load the dataset
file_path = "/mnt/data/Electric_Vehicle_Population_Data (1).csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# 2. Select relevant features (numeric + categorical)
# Drop columns that are identifiers or not useful for clustering
drop_cols = ['VIN (1-10)', 'DOL Vehicle ID', 'Vehicle Location']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 3. Handle missing values
df = df.dropna()

# 4. Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 5. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 6. Apply Agglomerative Clustering
# Change n_clusters as needed
agglo = AgglomerativeClustering(
    n_clusters=4,
    metric='euclidean',
    linkage='ward'
)

clusters = agglo.fit_predict(X_scaled)

# 7. Add cluster labels to dataframe
df['Cluster'] = clusters

print("\nCluster counts:")
print(df['Cluster'].value_counts())

# 8. Dimensionality reduction for visualization (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 9. Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=10)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Agglomerative Clustering of EV Dataset")
plt.colorbar(label='Cluster')
plt.show()
