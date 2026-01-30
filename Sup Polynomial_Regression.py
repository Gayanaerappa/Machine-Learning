import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv(r"C:\Users\gayan\OneDrive\Desktop\PYTHON VS CODE\Database\Electric_Vehicle_Population_Data.csv")
print(df.head())
X = df[['Model Year']]   # Independent variable
y = df['Electric Range']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
model = LinearRegression()
model.fit(X_poly_train, y_train)
y_pred = model.predict(X_poly_test)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
plt.scatter(X, y, color='blue', label="Actual Data")

X_line = np.linspace(X.min(), X.max(), 100)
X_line_poly = poly.transform(X_line)

plt.plot(X_line, model.predict(X_line_poly),
         color='red', linewidth=2, label="Polynomial Curve")

plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.title("Polynomial Regression")
plt.legend()
plt.show()