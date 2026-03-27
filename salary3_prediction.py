# Salary Prediction using Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression

# Sample Data (Experience vs Salary)
# X = Years of Experience
X = np.array([[1], [2], [3], [4], [5]])

# y = Salary
y = np.array([20000, 30000, 40000, 50000, 60000])

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Take user input
exp = float(input("Enter years of experience: "))

# Predict Salary
predicted_salary = model.predict([[exp]])

print(f"💰 Predicted Salary: {predicted_salary[0]:.2f}")
