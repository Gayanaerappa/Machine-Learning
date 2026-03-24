# Customer Segmentation using K-Means Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create Dataset (Annual Income & Spending Score)
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 20, 35, 36, 37, 38, 39, 40, 60, 61, 62, 63, 64, 65],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 35, 80, 5, 82, 41, 75, 50, 55, 52, 54, 53, 56]
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

# Step 2: Find optimal clusters using Elbow Method
wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 6), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Step 4: Visualization
plt.scatter(X['Annual_Income'], X['Spending_Score'], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X')

plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
