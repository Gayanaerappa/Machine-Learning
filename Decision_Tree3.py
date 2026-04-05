import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Dataset
data = {
    'Income': [25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],
    'Credit_Score': [600, 650, 700, 720, 750, 780, 800, 820],
    'Age': [22, 25, 30, 35, 40, 45, 50, 55],
    'Loan_Status': [0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Not Approved, 1 = Approved
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df[['Income', 'Credit_Score', 'Age']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Train Model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test with new data
new_data = [[55000, 730, 32]]
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Loan Approved ✅")
else:
    print("Loan Not Approved ❌")
