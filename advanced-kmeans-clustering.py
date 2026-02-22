import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset (no labels)
data = {
    "income": [25, 28, 30, 80, 82, 85, 40, 42],
    "spending": [30, 35, 32, 75, 78, 80, 50, 52],
    "age": [22, 25, 24, 45, 46, 48, 30, 31]
}

df = pd.DataFrame(data)

# K-Means model
model = KMeans(n_clusters=3, random_state=42)

# Train model
model.fit(df)

# Assign clusters
df["cluster"] = model.labels_

print(df)

# Visualization (Income vs Spending)
plt.scatter(df["income"], df["spending"], c=df["cluster"])
plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Customer Segmentation using K-Means")
plt.show()
