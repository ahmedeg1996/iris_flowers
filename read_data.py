import pandas as pd

dataset = pd.read_csv('iris_flowers.csv')

dataset = dataset.values

X = dataset[:,0:4]
y = dataset[:,5]

print(X)