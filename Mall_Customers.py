# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
# Example dataset: Mall Customers (you can download from Kaggle)
data = pd.read_csv("Mall_Customers.csv")

# Step 3: Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Find optimal K (Elbow Method)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 6: Train KMeans (choose K=5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 7: Visualization
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans)
plt.title("Customer Segments")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()
