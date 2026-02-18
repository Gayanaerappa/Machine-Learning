import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# 1. Dataset
# -----------------------------

data = {
    "Size": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5, 5],
    "Age": [20, 15, 18, 10, 8, 5, 3, 1],
    "Price": [2000000, 2500000, 3000000, 3800000,
              4500000, 5200000, 5800000, 6500000]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Features & Target
# -----------------------------

X = df[["Size", "Bedrooms", "Age"]]
y = df["Price"]

# -----------------------------
# 3. Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# 6. Accuracy (R² Score)
# -----------------------------

score = r2_score(y_test, y_pred)

print("\n✅ R² Score:", score)

# -----------------------------
# 7. Custom Prediction
# -----------------------------

print("\nEnter House Details:")

size = float(input("Size (sqft): "))
bedrooms = int(input("Bedrooms: "))
age = float(input("Age of House: "))

predicted_price = model.predict([[size, bedrooms, age]])

print(f"\n🏠 Predicted House Price: ₹{predicted_price[0]:.2f}")
