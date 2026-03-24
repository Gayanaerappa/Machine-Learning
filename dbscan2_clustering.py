# Customer Segmentation using DBSCAN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Step 1: Create Dataset
data = {
    'Annual_Income': [15, 16, 17, 18, 19, 20, 35, 36, 37, 38, 39, 40, 60, 61, 62, 63, 64, 65],
    'Spending_Score': [39, 81, 6, 77, 40, 76, 35, 80, 5, 82, 41, 75, 50, 55, 52, 54, 53, 56]
}

df = pd.DataFrame(data)

X = df[['Annual_Income', 'Spending_Score']]

# Step 2: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=2)
clusters = dbscan.fit_predict(X_scaled)

# Step 4: Add cluster labels
df['Cluster'] = clusters

# Step 5: Visualization
plt.scatter(X['Annual_Income'], X['Spending_Score'], c=clusters)

plt.title('DBSCAN Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# Step 6: Print results
print(df)
