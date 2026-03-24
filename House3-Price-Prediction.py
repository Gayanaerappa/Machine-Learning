# Simple Machine Learning Project
# House Price Prediction using Linear Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Step 1: Create Dataset
data = {
    'Area': [1000, 1500, 1800, 2400, 3000, 3500, 4000],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5],
    'Price': [300000, 400000, 450000, 600000, 700000, 800000, 900000]
}

df = pd.DataFrame(data)

# Step 2: Define Features and Target
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate Model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# Step 7: Test with new data
new_house = np.array([[2000, 3]])  # Area=2000, Bedrooms=3
prediction = model.predict(new_house)

print("Predicted Price for new house:", prediction[0])
