import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

X = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [10, 11, 12, 13, 14],
    [11, 12, 13, 14, 15]
])

tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X)

print("Reduced Data:")
print(X_reduced)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.show()
