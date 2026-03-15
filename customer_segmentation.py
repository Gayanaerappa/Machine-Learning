import pandas as pd
from sklearn.cluster import KMeans

# Sample Dataset
data = {
    "Annual_Income":[15,16,17,18,80,85,90,95],
    "Spending_Score":[39,81,6,77,40,76,6,94]
}

df = pd.DataFrame(data)

# K-Means Model
model = KMeans(n_clusters=2)

# Fit model
model.fit(df)

# Cluster prediction
clusters = model.predict(df)

df["Cluster"] = clusters

print(df)
