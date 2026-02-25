# Machine Learning Practice Project
# House Price Prediction using Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create Sample Dataset
data = {
    "Area": [1000, 1500, 1800, 2400, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Price": [3000000, 4500000, 5000000, 6500000, 8000000]
}

df = pd.DataFrame(data)

print("📊 Dataset:\n")
print(df)

# Step 2: Define Features & Target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Evaluate
error = mean_absolute_error(y_test, predictions)

print("\n🔮 Predictions:", predictions)
print("📉 Mean Absolute Error:", error)

# Step 7: Predict New House Price
new_house = [[2000, 3]]  # Area=2000, Bedrooms=3
predicted_price = model.predict(new_house)

print("\n🏠 Predicted Price for 2000 sqft, 3BHK:", predicted_price[0])
