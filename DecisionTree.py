import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#
#
data = pd.read_csv(r"D:\Datasets\village_130_1.csv")
# 将CF (Outcome Numeric)改名为Outcome并移动至最后一列
data['Outcome'] = data['CF (Outcome Numeric)']
data.drop(['CF (Outcome Numeric)'], axis=1, inplace=True)
#
# DROP
data.drop(['Row'], axis=1, inplace=True)
# data.drop(['Is Last Attempt'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
# data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['CF (Hiatus sec)'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
# data.drop(['CF (Student Chose Repeat)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
# data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Anon Student Id'], axis=1, inplace=True)
data.drop(['Duration (sec)'], axis=1, inplace=True)

print('\n\n\n\n保留的columns')
data.info()

X = data.iloc[:, data.columns != "Outcome"]
y = data.iloc[:, data.columns == "Outcome"]

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)


# 修正索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# 参数已调整
clf = DecisionTreeClassifier(random_state=50
                             , criterion='gini'
                             , max_depth=3
                             , min_impurity_decrease=0.0
                             , min_samples_leaf=1
                             , splitter='best')
clf = clf.fit(Xtrain, Ytrain)
score_ = clf.score(Xtest, Ytest)

print("\n\n\nscore")
print(score_)

score = cross_val_score(clf, X, y, cv=10).mean()
print("\nscore mean")
print(score)

# 拟合
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(
        random_state = 50
        , criterion = 'gini'
        , max_depth = 3
        , min_impurity_decrease = 0.0
        , min_samples_leaf = 1
        , splitter = 'best')
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te = cross_val_score(clf, X, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)

plt.plot(range(1, 11), tr, color="red", label="train")
plt.plot(range(1, 11), te, color="blue", label="test")
plt.xticks(range(1, 11))
plt.legend()
plt.show()
