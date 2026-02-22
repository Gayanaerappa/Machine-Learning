import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "pass_exam":     [0, 0, 0, 1, 1, 1, 1, 1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features & Target
X = df[["hours_studied"]]
y = df["pass_exam"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Model creation
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Predict new student
new_data = [[3.5]]
result = model.predict(new_data)

if result[0] == 1:
    print("\nPrediction: Student will PASS")
else:
    print("\nPrediction: Student will FAIL")
