import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -----------------------------
# 1. Create Small Dataset
# -----------------------------

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([10, 20, 30, 40, 50, 60])

# -----------------------------
# 2. Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# -----------------------------
# 3. Polynomial Transformation (Complex Model)
# -----------------------------

poly = PolynomialFeatures(degree=5)  # VERY HIGH DEGREE
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# -----------------------------
# 4. Train Model
# -----------------------------

model = LinearRegression()
model.fit(X_train_poly, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------

train_pred = model.predict(X_train_poly)
test_pred = model.predict(X_test_poly)

# -----------------------------
# 6. Accuracy Comparison
# -----------------------------

train_score = r2_score(y_train, train_pred)
test_score = r2_score(y_test, test_pred)

print("\n✅ Training R² Score:", train_score)
print("❌ Testing R² Score:", test_score)

# -----------------------------
# 7. Visualization
# -----------------------------

X_range = np.linspace(1, 6, 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

plt.figure()
plt.scatter(X, y)
plt.plot(X_range, y_range_pred)
plt.title("Overfitting Demonstration")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
