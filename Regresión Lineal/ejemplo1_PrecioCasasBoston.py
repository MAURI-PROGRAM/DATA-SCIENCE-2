#importar libreria de datos
from sklearn import datasets
#importar libreria de ML regresion lineal
from sklearn.linear_model import LinearRegression
#importar libreria para graficos de visualizacion
import matplotlib.pyplot as plt

import numpy as np

#asignar los datos a una variable
boston = datasets.load_boston()

#crear un objeto de regresion lineal
lr = LinearRegression(normalize=True)
#entrenar el objeto de regresion lineal con los datos obtenidos
#el primer argumento será toda las variables y la segunda el resultado
lr.fit(boston.data, boston.target)

#observar el coeficiente con respectos al valor de los precios
for (feature, coef) in zip(boston.feature_names, lr.coef_):
    print('{:>7}: {: 9.5f}'.format(feature, coef))

#funcion para observar mediante graficos las relaciones
def plot_feature(feature):
    f = (boston.feature_names == feature)
    plt.scatter(boston.data[:,f], boston.target, c='b', alpha=0.3)
    plt.plot(boston.data[:,f], boston.data[:,f]*lr.coef_[f] + lr.intercept_, 'k')
    plt.legend(['Predicted value', 'Actual value'])
    plt.xlabel(feature)
    plt.ylabel("Median value in $1000's")
    plt.show()
#llamada de la funcion
plot_feature('AGE')

#predecir el precio de una vivienda tras entrenarlo
predictions = lr.predict(boston.data)
f, ax = plt.subplots(1)
ax.hist(boston.target - predictions, bins=50, alpha=0.7)
ax.set_title('Histograma de residuales')
ax.text(0.95, 0.90, 'Media de residuales: {:.3e}'.format(np.mean(boston.target - predictions)),
        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
plt.show()

