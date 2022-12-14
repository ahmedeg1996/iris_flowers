import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('iris_flowers.csv')
dataset = dataset.values
X = dataset[:,0:4]
y = dataset[:,5]

scaling = StandardScaler().fit(X)
X_scaled= scaling.transform(X)

print(X_scaled)