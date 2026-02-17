import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (House Size → Price)
X = np.array([500, 800, 1000, 1200, 1500, 1800, 2000]).reshape(-1, 1)
y = np.array([1000000, 1500000, 2000000, 2300000, 3000000, 3500000, 4000000])

# Split data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print("Model Accuracy (R² Score):", score)

# Predict new house price
size = float(input("Enter house size: "))
predicted_price = model.predict([[size]])

print("Predicted House Price:", predicted_price[0])
