# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 4, 5, 7, 8, 10, 12, 13, 15, 16])

# Step 3: Manual Train-Test Split (80% train, 20% test)
X_train = X[:8]   # first 8 rows for training
y_train = y[:8]

X_test = X[8:]    # last 2 rows for testing
y_test = y[8:]

print("X_train:\n", X_train)
print("y_train:\n", y_train)
print("X_test:\n", X_test)
print("y_test:\n", y_test)

# Step 4: Create and Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on Test Data
y_pred = model.predict(X_test)

print("\nPredicted Salary for Test Data:", y_pred)

# Step 6: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Step 7: Visualize Regression Line
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression: Salary vs Experience")
plt.legend()
plt.show()

# Optional: See slope and intercept
print("\nSlope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
