import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
dataset = pd.read_csv('iris_flowers.csv')
dataset = dataset.values
X = dataset[:,0:4]
y = dataset[:,5]

scaling = StandardScaler().fit(X)
X_scaled= scaling.transform(X)
labeling = LabelEncoder().fit(y)
y_labled = labeling.transform(y)

model = Ridge()

kfold = KFold(10 , random_state = 7 , shuffle = True)

val_score = cross_val_score(model , X_scaled , y_labled , cv = kfold)
print(val_score)
print(val_score.mean())