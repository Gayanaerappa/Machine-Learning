import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Dataset
data = {
    'X': [1, 2, 3, 8, 9, 10, 25, 26, 27],
    'Y': [2, 3, 4, 9, 10, 11, 30, 31, 32]
}

df = pd.DataFrame(data)

# Step 2: Elbow Method
wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 6), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Step 3: Apply KMeans (choose k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Step 4: Print Output
print(df)

# Step 5: Visualization
plt.scatter(df['X'], df['Y'], c=df['Cluster'])
plt.title("K-Means Clustering")
plt.show()
