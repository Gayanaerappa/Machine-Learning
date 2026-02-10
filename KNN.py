import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Input data: hours studied
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# Output data: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create KNN model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, y)

# Predict for new student
hours = [[4]]
prediction = model.predict(hours)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")
