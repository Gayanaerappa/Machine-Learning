# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset
# X = Years of Experience
# y = Salary in Lakhs
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 4, 5, 7, 8, 10, 12, 13, 15, 16])

# Step 3: Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Model
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Predict on Test Data
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Salary for Test Data:", y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 8: Visualize
plt.scatter(X, y, color='blue')         # Actual data
plt.plot(X, model.predict(X), color='red')  # Regression line
plt.title("Linear Regression: Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (Lakhs)")
plt.show()
