# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create sample customer data
data = {
    "Annual_Income": [15, 16, 17, 18, 19, 60, 62, 65, 70, 72],
    "Spending_Score": [39, 81, 6, 77, 40, 55, 60, 61, 65, 70]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Select features
X = df[['Annual_Income', 'Spending_Score']]

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Predict clusters
df['Cluster'] = kmeans.labels_

print(df)

# Plot clusters
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
