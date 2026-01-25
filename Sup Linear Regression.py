
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
df = pd.read_csv(r"C:\Users\gayan\OneDrive\Desktop\PYTHON VS CODE\Database\Electric_Vehicle_Population_Data.csv")
print(df.head())
# 3. Select required columns
# Input (X) -> Model Year
# Output (y) -> Electric Range
X = df[['Model Year']]
y = df['Electric Range']
# 4. Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 5. Create Linear Regression Model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)
# 8. Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# 9. Print results
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", r2)
print("RMSE:", rmse)
# 10. Visualization
plt.scatter(X_test, y_test, color='blue', label="Actual Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.title("Linear Regression: Model Year vs Electric Range")
plt.legend()
plt.show()