import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Dataset
data = {
    'Experience': [1, 2, 3, 5, 7, 10, 12, 15],
    'Education_Level': [1, 1, 2, 2, 3, 3, 3, 3],  # 1=UG, 2=PG, 3=PhD
    'Working_Hours': [6, 7, 8, 8, 9, 9, 10, 10],
    'Salary': [0, 0, 0, 1, 1, 1, 1, 1]  # 0=Low, 1=High
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df[['Experience', 'Education_Level', 'Working_Hours']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Train Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test with new data
new_employee = [[6, 2, 9]]  # 6 years exp, PG, 9 hours
prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("High Salary 💰")
else:
    print("Low Salary 📉")
