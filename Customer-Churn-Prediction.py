# Machine Learning Project: Customer Churn Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("churn_data.csv")

# Remove unnecessary column
if 'customerID' in data.columns:
    data.drop(columns='customerID', inplace=True)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical to numeric
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop(columns='Churn', axis=1)
y = data['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Custom Prediction
input_data = X.iloc[0].values.reshape(1, -1)
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Customer will leave")
else:
    print("Customer will stay")
