from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# House size (square feet)
X = [[500], [700], [900], [1100], [1300], [1500], [1700]]

# Rent price
y = [8000, 10000, 12000, 15000, 17000, 20000, 23000]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict rent for test data
prediction = model.predict(X_test)

print("House Size (Test):", X_test)
print("Actual Rent:", y_test)
print("Predicted Rent:", prediction)

# Predict rent for 2000 sq ft house
new_rent = model.predict([[2000]])
print("Predicted Rent for 2000 sq ft house:", new_rent)
