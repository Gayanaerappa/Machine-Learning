# Step 1: Import Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Create Sample Dataset (Study Hours vs Pass/Fail)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  
# 0 = Fail, 1 = Pass

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Create Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Check Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict New Student
new_hours = np.array([[5]])
prediction = model.predict(new_hours)

if prediction[0] == 1:
    print("Student will Pass ✅")
else:
    print("Student will Fail ❌")
