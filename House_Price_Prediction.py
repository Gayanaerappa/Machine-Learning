import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Dataset
data = {
    'Area': [500, 800, 1000, 1200, 1500, 1800, 2000, 2200],
    'Price': [100, 150, 200, 250, 300, 350, 400, 450]
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df[['Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Step 6: Test with new data
new_area = [[1600]]
predicted_price = model.predict(new_area)

print("Predicted Price:", predicted_price[0])
