from sklearn.linear_model import LinearRegression

# Input: Hours studied
X = [[1],[2],[3],[4],[5]]

# Output: Marks
y = [35,45,55,65,75]

model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 hours
print(model.predict([[6]]))