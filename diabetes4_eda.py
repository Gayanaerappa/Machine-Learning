# Diabetes EDA Project

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset (you can create or download CSV)
# Sample dataset creation
data = {
    "Glucose": [85, 89, 90, 120, 140, 160, 130, 150],
    "BMI": [22, 25, 28, 30, 35, 32, 31, 29],
    "Outcome": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Show first rows
print("📌 First 5 rows:")
print(df.head())

# Summary statistics
print("\n📊 Data Summary:")
print(df.describe())

# Outcome count plot
df["Outcome"].value_counts().plot(kind="bar")
plt.title("Diabetes Outcome Count")
plt.xlabel("Outcome (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Histogram for Glucose
plt.hist(df["Glucose"])
plt.title("Glucose Distribution")
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.show()
