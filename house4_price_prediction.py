# House Price Prediction using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    'Area': [500, 800, 1000, 1200, 1500, 1800, 2000],
    'Price': [1500000, 2400000, 3000000, 3600000, 4500000, 5400000, 6000000]
}

df = pd.DataFrame(data)

# Step 2: Split input and output
X = df[['Area']]
y = df['Price']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict
predicted_price = model.predict([[1600]])
print("Predicted price for 1600 sq.ft:", int(predicted_price[0]))

# Step 5: Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.legend()
plt.show()
