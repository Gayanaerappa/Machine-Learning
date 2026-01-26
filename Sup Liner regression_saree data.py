import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
df =pd.read_excel(r"C:\Users\gayan\OneDrive\Desktop\PYTHON VS CODE\Database\saree_sales_100rows.xlsx")
print(df.head())
x = df[["Quantity_Sold", "Cost_Price", "Selling_Price", "Total_Expenses"]]
y = df["Profit"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=42 )
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
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
plt.scatter(x_test["Quantity_Sold"], y_test)
plt.scatter(x_test["Quantity_Sold"], y_pred)
plt.xlabel("Quantity Sold")
plt.ylabel("Profit")
plt.title("Quantity Sold vs Profit")
plt.show()
plt.plot(x_test["Quantity_Sold"], y_test)
plt.plot(x_test["Quantity_Sold"], y_pred)
plt.xlabel("Quantity Sold")
plt.ylabel("Profit")
plt.title("Actual vs Predicted Profit")
plt.show()