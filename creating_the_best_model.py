import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
dataset = pd.read_csv('iris_flowers.csv')

dataset = dataset.values

X = dataset[:,0:4]
y = dataset[:,5]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
labeling = LabelEncoder().fit(y)
y_labled = labeling.transform(y)

X_train , X_test , y_train , y_test = train_test_split(X_scaled , y_labled , random_state = 42 , shuffle = True)

model = LinearDiscriminantAnalysis()

model = model.fit(X_train , y_train)

file_name = 'iris_flowers_model'
pickle.dump(model , open(file_name , 'wb'))

# testing the model

my_model = pickle.load(open('iris_flowers_model' , 'rb'))

print(my_model.score(X_test))
