import pandas as pd
from sklearn.cluster import KMeans

# Sample dataset (no labels)
data = {
    "hours_studied": [1, 2, 3, 6, 7, 8],
    "exam_score":    [30, 35, 40, 75, 80, 85]
}

df = pd.DataFrame(data)

# Model creation (choose number of clusters)
model = KMeans(n_clusters=2, random_state=42)

# Train model
model.fit(df)

# Cluster assignment
df["cluster"] = model.labels_

print(df)
