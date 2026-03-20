# House Price Prediction Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create Dataset
data = {
    'Area': [500, 800, 1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000],
    'Bedrooms': [1, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    'Age': [10, 8, 6, 5, 4, 3, 3, 2, 1, 1],
    'Price': [50, 80, 120, 150, 180, 200, 220, 250, 270, 300]
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Prediction
y_pred = model.predict(X_test)

# Step 5: Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 6: User Input
area = float(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Bedrooms: "))
age = int(input("Enter Age of house: "))

prediction = model.predict([[area, bedrooms, age]])

print(f"Estimated House Price: {prediction[0]:.2f} Lakhs")

# Step 7: Visualization (Area vs Price)
plt.scatter(df['Area'], df['Price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()
