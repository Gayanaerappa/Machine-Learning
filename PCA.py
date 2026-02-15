import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.array([
    [2, 3, 4],
    [3, 4, 5],
    [5, 6, 7],
    [8, 9, 10]
])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Reduced Data:")
print(X_reduced)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()
