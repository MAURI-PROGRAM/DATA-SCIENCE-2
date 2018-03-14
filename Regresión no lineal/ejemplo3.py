#importar libreria para graficos de visualizacion
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

alphas = np.linspace(0.01, 0.5)
f, ax = plt.subplots()
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
# añadimos algo de ruido
xr = x + np.random.normal(scale=0.1, size=x.shape)
yr = y + np.random.normal(scale=0.2, size=y.shape)
ax.plot(x, np.sin(x), 'r', label='sin ruido')
ax.scatter(xr, yr, label='con ruido')
# convertimos nuestro array en un vector columna
X = xr[:, np.newaxis]
# utilizamos un bucle para probar polinomios de diferente grado
for degree in [3, 4, 5]:
    # utilizamos Pipeline para crear una secuencia de pasos
    model = make_pipeline(PolynomialFeatures(degree), RidgeCV(alphas=alphas))
    model.fit(X, y)
    y = model.predict(x[:, np.newaxis])
    ax.plot(x, y, '--', lw=2, label="degree %d" % degree)
ax.legend()
plt.show()