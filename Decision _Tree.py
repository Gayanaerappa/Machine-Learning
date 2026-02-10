import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Input: hours studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = DecisionTreeClassifier()

# Train
model.fit(X, y)

# Predict
result = model.predict([[3.5]])

print("PASS" if result[0] == 1 else "FAIL")
🧠 How it thinks
