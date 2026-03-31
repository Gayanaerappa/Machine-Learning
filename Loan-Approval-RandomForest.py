# Machine Learning Project: Loan Approval Prediction

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("loan_data.csv")

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical to numeric
le = LabelEncoder()

columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in columns:
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = data['Loan_Status']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Custom Prediction
input_data = (1, 1, 0, 0, 5000, 2000, 150, 360, 1, 2)

input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
