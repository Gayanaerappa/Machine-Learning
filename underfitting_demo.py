import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# 1. Create Non-Linear Dataset
# -----------------------------

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])  # Quadratic pattern (y = x²)

# -----------------------------
# 2. Train Simple Linear Model
# -----------------------------

model = LinearRegression()
model.fit(X, y)

# -----------------------------
# 3. Predictions
# -----------------------------

y_pred = model.predict(X)

# -----------------------------
# 4. Accuracy
# -----------------------------

score = r2_score(y, y_pred)

print("\n❌ R² Score (Underfitting Example):", score)

# -----------------------------
# 5. Visualization
# -----------------------------

plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Underfitting Demonstration")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
