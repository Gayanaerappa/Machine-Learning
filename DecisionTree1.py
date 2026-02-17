import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_model():
    # Training data (Hours Studied)
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

    # Labels (0 = Fail, 1 = Pass)
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model

def predict_result(model):
    hours = float(input("Enter study hours: "))
    result = model.predict([[hours]])

    if result[0] == 1:
        print(" Student will PASS")
    else:
        print("Student will FAIL")

# Train + Predict
ml_model = train_model()
predict_result(ml_model)
