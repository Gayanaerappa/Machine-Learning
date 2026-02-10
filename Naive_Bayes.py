import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])

model = GaussianNB()
model.fit(X, y)

pred = model.predict([[4]])
print("PASS" if pred[0] == 1 else "FAIL")
