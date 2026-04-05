import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Dataset
data = {
    'Age': [22, 25, 30, 35, 40, 45, 50, 55],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    'Purchased': [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 2: Split Data
X = df[['Age', 'Salary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 3: Train Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Test with new data
new_customer = [[33, 32000]]
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Customer will BUY 🛒")
else:
    print("Customer will NOT BUY ❌")
