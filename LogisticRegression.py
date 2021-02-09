import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"/Users/song/Desktop/village_130_1.csv")  # , index_col=0)
data.info()

pd.set_option('display.max_columns', 5)
pd.set_option('display.expand_frame_repr', False)
corr = data[['Duration (sec)', 'CF (Matrix)', 'Attempt At Step', 'Is Last Attempt', 'CF (Attempt Number)',
             'CF (Student Chose Repeat)', 'CF (Student Used Scaffold)']].corr()
print(corr)

heat_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

# To get a uniform distributed data set for each person
x_train_set = list()
x_test_set = list()
y_train_set = list()
y_test_set = list()
for i in data['Anon Student Id'].unique():
    temp_data = data[data['Anon Student Id'] == i]
    x = temp_data[['Duration (sec)', 'CF (Matrix)', 'Attempt At Step', 'Is Last Attempt', 'CF (Attempt Number)',
                   'CF (Student Chose Repeat)', 'CF (Student Used Scaffold)']]
    y = temp_data['CF (Outcome Numeric)']
    temp_x_train, temp_x_test, temp_y_train, temp_y_test = train_test_split(x, y, test_size=0.2, random_state=10)
    x_train_set.append(temp_x_train)
    y_train_set.append(temp_y_train)
    x_test_set.append(temp_x_test)
    y_test_set.append(temp_y_test)
x_train = pd.concat(x_train_set)
x_test = pd.concat(x_test_set)
y_train = pd.concat(y_train_set)
y_test = pd.concat(y_test_set)

logit = LogisticRegression()
logit.fit(x_train, y_train)

predict = logit.predict(x_test)
print('predict', predict)
Score = accuracy_score(y_test, predict)

print(Score)
