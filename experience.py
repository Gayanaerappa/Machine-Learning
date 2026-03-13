from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Years of experience
X = [[1], [2], [3], [4], [5], [6], [7]]

# Salary
y = [20000, 25000, 30000, 35000, 40000, 45000, 50000]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict salary for test data
prediction = model.predict(X_test)

print("Test Experience:", X_test)
print("Actual Salary:", y_test)
print("Predicted Salary:", prediction)

# Predict new value
new_salary = model.predict([[8]])
print("Salary for 8 years experience:", new_salary)
