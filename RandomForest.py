import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv('Electric_Vehicle_Population_Data (1).csv')

# 2. Data Preprocessing
# Selecting relevant features for prediction
# We'll predict 'Electric Vehicle Type'
features = ['Model Year', 'Make', 'Model', 'Electric Range', 'Base MSRP', 'County']
target = 'Electric Vehicle Type'

# Dropping rows with missing values in our selected columns
data = df[features + [target]].dropna()

# Encoding categorical variables (Random Forest needs numerical input)
le_make = LabelEncoder()
le_model = LabelEncoder()
le_county = LabelEncoder()
le_target = LabelEncoder()

data['Make'] = le_make.fit_transform(data['Make'])
data['Model'] = le_model.fit_transform(data['Model'])
data['County'] = le_county.fit_transform(data['County'])
data[target] = le_target.fit_transform(data[target])

# 3. Splitting the Data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training the Random Forest Classifier
# n_estimators=100 means we are building 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predictions and Evaluation
y_pred = rf_model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 6. Feature Importance Visualization
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names, palette='viridis')
plt.title('Feature Importance in Predicting EV Type')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Output a confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
