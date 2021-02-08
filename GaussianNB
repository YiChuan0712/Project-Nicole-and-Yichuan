import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

import pandas as pd


data = pd.read_csv(r"D:\Datasets\village_130_1.csv")#, index_col=0)
data['Outcome'] = data['CF (Outcome Numeric)']
data.drop(['CF (Outcome Numeric)'], axis=1, inplace=True)

data.drop(['Row'], axis=1, inplace=True)
# data.drop(['Is Last Attempt'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
# data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['CF (Hiatus sec)'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
#data.drop(['CF (Student Chose Repeat)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
# data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Anon Student Id'], axis=1, inplace=True)
data.drop(['Duration (sec)'], axis=1, inplace=True)
data.info()

X = data.iloc[:, data.columns != "Outcome"]
y = data.iloc[:, data.columns == "Outcome"]

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

gnb = GaussianNB().fit(Xtrain,Ytrain)
acc_score = gnb.score(Xtest,Ytest)

print("\n\n\nscore")
print(acc_score)

score = cross_val_score(gnb, X, y, cv=10).mean()
print("\nscore mean")
print(score)
