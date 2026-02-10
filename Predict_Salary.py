import numpy as np
from sklearn.linear_model import LinearRegression

# Input data (Experience in years)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data (Salary)
y = np.array([15000, 20000, 25000, 30000, 35000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict salary for 6 years experience
experience = [[6]]
predicted_salary = model.predict(experience)

print("Predicted Salary:", predicted_salary[0])
