import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 2, 6],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 85, 58, 78],
    'Result': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df[['Hours_Studied', 'Attendance']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: KNN Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 4: Prediction
y_pred = model.predict(X_test)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test with new data
new_data = [[4, 65]]
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Student will PASS 🎉")
else:
    print("Student will FAIL ❌")
