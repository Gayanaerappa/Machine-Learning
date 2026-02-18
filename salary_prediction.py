import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Sample Dataset
# -----------------------------

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [25000, 28000, 32000, 38000, 42000,
               50000, 54000, 62000, 70000, 78000]
}

df = pd.DataFrame(data)

print("\nDataset:")
print(df)

# -----------------------------
# 2. Features & Target
# -----------------------------

X = df[["Experience"]]   # Independent Variable
y = df["Salary"]         # Dependent Variable

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

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

# -----------------------------
# 6. Custom User Prediction
# -----------------------------

exp = float(input("\nEnter Years of Experience: "))
predicted_salary = model.predict([[exp]])

print(f"💰 Predicted Salary: ₹{predicted_salary[0]:.2f}")

# -----------------------------
# 7. Visualization
# -----------------------------

plt.figure()
plt.scatter(df["Experience"], df["Salary"])
plt.plot(df["Experience"], model.predict(X),)
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Model")
plt.show()
