# Sales Data Analysis Project

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create Dataset
data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 130, 250, 300],
    'Revenue': [1000, 1500, 2000, 1300, 2500, 3000]
}

df = pd.DataFrame(data)

# Step 2: Basic Info
print("Dataset:\n", df)
print("\nSummary:\n", df.describe())

# Step 3: Total Revenue
total_revenue = df['Revenue'].sum()
print("\nTotal Revenue:", total_revenue)

# Step 4: Best Month
best_month = df.loc[df['Sales'].idxmax()]
print("\nBest Month:\n", best_month)

# Step 5: Product-wise Sales
product_sales = df.groupby('Product')['Sales'].sum()
print("\nProduct Sales:\n", product_sales)

# Step 6: Visualization

# Bar Chart - Sales per Month
plt.figure()
plt.bar(df['Month'], df['Sales'])
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Pie Chart - Product Distribution
plt.figure()
plt.pie(product_sales, labels=product_sales.index, autopct='%1.1f%%')
plt.title("Product Sales Distribution")
plt.show()
