import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create Dataset
data = {
    "Size_sqft": [500, 800, 1000, 1200, 1500, 1800],
    "Price": [1000000, 1500000, 2000000, 2400000, 3000000, 3500000]
}

df = pd.DataFrame(data)

# Step 2: Define X and y
X = df[["Size_sqft"]]   # Independent variable
y = df["Price"]         # Dependent variable

# Step 3: Create Model
model = LinearRegression()

# Step 4: Train Model
model.fit(X, y)

# Step 5: Predict New House Price
new_size = [[2000]]
predicted_price = model.predict(new_size)

print("Predicted Price for 2000 sqft house:", predicted_price[0])

# Step 6: Plot Graph
plt.scatter(df["Size_sqft"], df["Price"])
plt.plot(df["Size_sqft"], model.predict(X))
plt.xlabel("House Size (sqft)")
plt.ylabel("Price")
plt.title("Linear Regression - House Price Prediction")
plt.show()
