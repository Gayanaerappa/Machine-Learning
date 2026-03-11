import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../data/salary_data.csv')

# Scatter plot
sns.scatterplot(data=data, x='YearsExperience', y='Salary')
plt.title('YearsExperience vs Salary')
plt.show()

# Fit model
model = LinearRegression()
model.fit(data[['YearsExperience']], data['Salary'])

# Plot regression line
plt.scatter(data['YearsExperience'], data['Salary'], color='blue')
plt.plot(data['YearsExperience'], model.predict(data[['YearsExperience']]), color='red')
plt.title('Linear Regression Line')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
