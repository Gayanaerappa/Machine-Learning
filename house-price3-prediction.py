import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Size_sqft": [600, 800, 1000, 1200, 1500, 1800],
    "Price": [1200000, 1500000, 2000000, 2300000, 3000000, 3500000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Size_sqft"]]
y = df["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict([[1400]])

print("Predicted House Price:", prediction)
