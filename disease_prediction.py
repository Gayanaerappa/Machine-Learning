# Disease Prediction Project
# Beginner Friendly Machine Learning Example

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Create Sample Dataset
# -----------------------------

data = {
    'Glucose': [85, 89, 90, 120, 140, 150, 95, 100, 130, 160],
    'BloodPressure': [66, 70, 80, 75, 90, 85, 72, 68, 88, 92],
    'BMI': [26.6, 28.1, 25.0, 30.5, 33.6, 35.2, 27.8, 24.5, 31.0, 36.5],
    'Age': [31, 29, 35, 40, 50, 55, 33, 28, 45, 60],
    'Disease': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df)

# -----------------------------
# 2. Split Features & Target
# -----------------------------

X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Disease']

# -----------------------------
# 3. Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Model
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Custom User Prediction
# -----------------------------

print("\nEnter Patient Details:")

glucose = float(input("Glucose Level: "))
bp = float(input("Blood Pressure: "))
bmi = float(input("BMI: "))
age = float(input("Age: "))

new_data = np.array([[glucose, bp, bmi, age]])

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\n⚠️ Prediction: Patient likely has the disease.")
else:
    print("\n✅ Prediction: Patient likely does NOT have the disease.")
