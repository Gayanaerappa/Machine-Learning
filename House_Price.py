import numpy as np
from sklearn.linear_model import LinearRegression

# Input data: House size (sqft)
X = np.array([500, 800, 1000, 1200, 1500]).reshape(-1, 1)

# Output data: House price
y = np.array([1000000, 1500000, 2000000, 2300000, 3000000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for new house
size = [[1100]]
predicted_price = model.predict(size)

print("Predicted House Price:", predicted_price[0])
