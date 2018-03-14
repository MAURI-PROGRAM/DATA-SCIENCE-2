from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# con el parámetro random_state nos aseguramos obtener siempre lo mismo
X, y = make_classification(n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1,
                           class_sep=0.9, random_state=27)

lr = LogisticRegression()
lr.fit(X, y)

plt.scatter(X, y, alpha=0.4, label='real')
plt.plot(np.sort(X, axis=0), lr.predict(np.sort(X, axis=0)), color='r', label='categoría')
plt.legend(loc=2)
plt.xlabel('X')
plt.ylabel('Probabilidad')

plt.show()