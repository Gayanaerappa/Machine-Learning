# Diabetes Prediction using Logistic Regression

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create dataset
data = {
    'Glucose': [85, 89, 78, 120, 130, 95, 140, 150],
    'BMI': [22.0, 26.5, 23.1, 30.2, 32.5, 28.0, 35.0, 36.5],
    'Age': [25, 30, 22, 45, 50, 35, 55, 60],
    'Outcome': [0, 0, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df[['Glucose', 'BMI', 'Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 3: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Test with new data
sample = [[120, 28.0, 40]]
prediction = model.predict(sample)

if prediction[0] == 1:
    print("Diabetes: YES")
else:
    print("Diabetes: NO")
