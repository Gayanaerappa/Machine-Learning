import numpy as np
from sklearn.svm import SVC

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])

model = SVC(kernel="linear")
model.fit(X, y)

pred = model.predict([[5]])
print("PASS" if pred[0] == 1 else "FAIL")
