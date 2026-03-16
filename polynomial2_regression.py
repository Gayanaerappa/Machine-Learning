# Polynomial Regression Example
# Predict Salary based on Experience

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 1: Dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([2,4,5,7,8,10,12,13,15,18])

# Step 2: Create Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Step 3: Train Model
model = LinearRegression()
model.fit(X_poly, y)

# Step 4: Predict
predicted_salary = model.predict(poly.transform([[11]]))
print("Predicted Salary for 11 years experience:", predicted_salary)

# Step 5: Visualization
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X_poly), color='red', label="Polynomial Curve")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression Example")
plt.legend()
plt.show() (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
