import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Engine_Size":[1000,1200,1500,1800,2000,2200],
    "Price":[300000,350000,500000,650000,750000,900000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["Engine_Size"]]
y = df["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict price for engine size 1600
prediction = model.predict([[1600]])

print("Predicted Car Price:", prediction)
