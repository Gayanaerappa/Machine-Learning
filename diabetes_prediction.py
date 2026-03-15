import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "Glucose":[85, 89, 90, 120, 140, 150, 160, 180],
    "BMI":[22, 25, 28, 30, 32, 35, 38, 40],
    "Age":[25, 30, 35, 40, 45, 50, 55, 60],
    "Diabetes":[0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Glucose","BMI","Age"]]
y = df["Diabetes"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("Prediction:", prediction)
print("Accuracy:", accuracy)
