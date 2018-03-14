#importar libreria para graficos de visualizacion
import matplotlib.pyplot as plt
import numpy as np


f, ax = plt.subplots()
x = np.linspace(0, 2*np.pi)
y = np.sin(x)
ax.plot(x, np.sin(x), 'r', label='sin ruido')
# añadimos algo de ruido
xr = x + np.random.normal(scale=0.1, size=x.shape)
yr = y + np.random.normal(scale=0.2, size=y.shape)
ax.scatter(xr, yr, label='con ruido')
ax.legend()
plt.show()