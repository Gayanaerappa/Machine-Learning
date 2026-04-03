# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")

# Split data
X = data[['hours', 'attendance']]
y = data['result']

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[5, 78]])
print("Prediction (1=Pass, 0=Fail):", prediction[0])

# Visualize Decision Tree
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=['hours', 'attendance'], class_names=['Fail', 'Pass'], filled=True)
plt.title("Decision Tree")
plt.show()
