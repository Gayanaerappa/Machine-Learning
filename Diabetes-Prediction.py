# Machine Learning Project: Diabetes Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")

# Split features and target
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
X_test_prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, X_test_prediction)
print("Accuracy:", accuracy)

# Test with custom input
input_data = (5,116,74,0,0,25.6,0.201,30)

input_array = np.asarray(input_data)
input_reshaped = input_array.reshape(1, -1)

input_scaled = scaler.transform(input_reshaped)

prediction = model.predict(input_scaled)

if prediction[0] == 0:
    print("Person is NOT diabetic")
else:
    print("Person is diabetic")
