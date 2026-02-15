import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [8, 8],
    [9, 9],
    [10, 8]
])

linked = linkage(X, method='ward')
dendrogram(linked)
plt.show()

hc = AgglomerativeClustering(n_clusters=2)
labels = hc.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
