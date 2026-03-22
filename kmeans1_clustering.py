# 🛍️ Customer Segmentation using K-Means Clustering

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample dataset (you can replace with CSV later)
data = {
    "Annual Income": [15, 16, 17, 18, 19, 40, 42, 44, 46, 48, 60, 62, 64, 66, 68],
    "Spending Score": [39, 81, 6, 77, 40, 50, 60, 55, 65, 70, 20, 25, 30, 35, 40]
}

df = pd.DataFrame(data)

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 🔍 Elbow Method to find optimal clusters
inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# 🎯 Apply K-Means (choose k=3 based on elbow)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels
df["Cluster"] = clusters

print("\n📊 Clustered Data:")
print(df)

# 📊 Visualization
plt.figure()
plt.scatter(df["Annual Income"], df["Spending Score"], c=clusters)
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
