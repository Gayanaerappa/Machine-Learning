from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Study hours (Input feature)
X = [[1], [2], [3], [4], [5], [6], [7], [8]]

# Marks (Target)
y = [35, 40, 50, 65, 70, 75, 85, 95]

# Split data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict using test data
prediction = model.predict(X_test)

print("Test Input:", X_test)
print("Actual Marks:", y_test)
print("Predicted Marks:", prediction)
