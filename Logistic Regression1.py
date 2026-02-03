import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\gayan\OneDrive\Desktop\PYTHON VS CODE\Database\Electric_Vehicle_Population_Data.csv")

# -------------------------------
# 2. Select Required Columns
# -------------------------------
df = df[['Model Year', 'Electric Range', 'Base MSRP',
         'Electric Vehicle Type']]

# Drop missing values
df.dropna(inplace=True)

# -------------------------------
# 3. Encode Target Variable
# -------------------------------
# BEV -> 1, PHEV -> 0
df['EV_Type'] = df['Electric Vehicle Type'].map({
    'Battery Electric Vehicle (BEV)': 1,
    'Plug-in Hybrid Electric Vehicle (PHEV)': 0
})

df.drop('Electric Vehicle Type', axis=1, inplace=True)

# -------------------------------
# 4. Split Features & Target
# -------------------------------
X = df.drop('EV_Type', axis=1)
y = df['EV_Type']

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 7. Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# 8. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9. Evaluation
# -------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))