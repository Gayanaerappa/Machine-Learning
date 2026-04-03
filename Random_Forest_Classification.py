# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Split data
X = data[['age', 'experience']]
y = data['salary']

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[30, 6]])
print("Prediction (1=High Salary, 0=Low Salary):", prediction[0])

# Feature Importance
importance = model.feature_importances_
print("Feature Importance:", importance)
