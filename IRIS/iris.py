from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = pd.read_csv('iris.csv')


# Ejemplo pairplot con datase iris
g = sns.pairplot(iris, hue='Species', diag_kind="hist")
plt.show()
