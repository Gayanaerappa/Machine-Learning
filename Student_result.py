import numpy as np
from sklearn.linear_model import LogisticRegression

# Input data: Hours studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output data: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create model
model = LogisticRegression()

# Train model
model.fit(X, y)

# Predict result for a student who studied 4.5 hours
hours = [[4.5]]
result = model.predict(hours)

if result[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")
