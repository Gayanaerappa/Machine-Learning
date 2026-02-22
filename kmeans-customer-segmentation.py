import pandas as pd
from sklearn.cluster import KMeans

# Sample dataset (no labels)
data = {
    "annual_income": [15, 16, 17, 80, 82, 85],
    "spending_score": [20, 22, 25, 75, 78, 80]
}

df = pd.DataFrame(data)

# Create K-Means model
model = KMeans(n_clusters=2, random_state=42)

# Train model
model.fit(df)

# Assign clusters
df["cluster"] = model.labels_

print(df)
