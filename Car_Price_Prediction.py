from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Car age (years)
X = [[1], [2], [3], [4], [5], [6], [7]]

# Car price (in rupees)
y = [800000, 750000, 700000, 650000, 600000, 550000, 500000]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices for test data
prediction = model.predict(X_test)

print("Car Age (Test Data):", X_test)
print("Actual Price:", y_test)
print("Predicted Price:", prediction)

# Predict price for new car age
new_price = model.predict([[8]])
print("Predicted price for 8-year-old car:", new_price)
