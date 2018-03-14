
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
#importar libreria para graficos de visualizacion
import matplotlib.pyplot as plt
import numpy as np

reg_data, reg_target = make_regression(n_samples=2000, n_features=3, effective_rank=2, noise=10)

# creamos un numpy array con los valores de alpha que queremos evaluar
alphas = np.linspace(0.01, 0.5)
# que pasamos a nuestro modelo RidgeCV, guardando los resultados
rcv = RidgeCV(alphas=alphas, store_cv_values=True)
rcv.fit(reg_data, reg_target)
# representamos gráficamente el error en función de alpha
plt.rc('text', usetex=False)
f, ax = plt.subplots()
ax.plot(alphas, rcv.cv_values_.mean(axis=0))
ax.text(0.05, 0.90, 'alpha que minimiza el error: {:.3f}'.format(rcv.alpha_),
        transform=ax.transAxes)
plt.show()