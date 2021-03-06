# 逻辑回归
#
#
# 下面是一部分导入的包 剩下的包在使用时才导入（尽管这并不是一个良好的代码习惯）
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
#
#
#


def aic(y, y_pred, p):
    n = len(y)
    resid = np.subtract(y_pred, y)
    rss = np.sum(np.power(resid, 2))
    aic_score = n * np.log(rss / n) + 2 * p
    return aic_score
#
def bic(y, y_pred, p):
    n = len(y)
    residual = np.subtract(y_pred, y)
    SSE = np.sum(np.power(residual, 2))
    BIC = n*np.log(SSE/n) + p*np.log(n)
    return BIC
#
# 导入.csv文件
data = pd.read_csv(r"/Users/LingZhang/Desktop/final_clean.csv")  # , index_col=0)
data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
data.drop(['CF (Matrix)_stories'], axis=1, inplace=True)

#
#
#
pd.set_option('display.max_columns', 5)
pd.set_option('display.expand_frame_repr', False)
data.info()
corr = data[['CF (Duration sec)', 'CF (Matrix Level)', 'Level (Tutor Name)_akira', 'Level (Tutor Name)_bigmath', 'Level (Tutor Name)_bubble_pop' ,'Level (Tutor Name)_picmatch',
             'Level (Tutor Name)_spelling','Level (Tutor Name)_story_questions','Level (Tutor Name)_story_reading','Level (Tutor Name)_word_copy',
             'CF (Matrix)_literacy','CF (Matrix)_math']].corr()
print(corr)
heat_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
# data.info()
X = data.iloc[:, data.columns != "CF (Outcome Numeric)"]
y = data.iloc[:, data.columns == "CF (Outcome Numeric)"]
#
#
#


list = data['Anon Student Id'].unique()
# print(list)
#
print("逻辑回归")
print("\n\n\n\n以下是每次踢出一个学生的交叉验证方法")
sum = []
for i in list:
    Test = data[data['Anon Student Id'].isin([i])].copy()
    Train = data[~data['Anon Student Id'].isin([i])].copy()
    Test.drop(['Anon Student Id'], axis=1, inplace=True)
    Train.drop(['Anon Student Id'], axis=1, inplace=True)
    # Test.info()
    # Train.info()
    Xtest = Test.iloc[:, Test.columns != "CF (Outcome Numeric)"]
    Ytest = Test.iloc[:, Test.columns == "CF (Outcome Numeric)"]
    Xtrain = Train.iloc[:, Train.columns != "CF (Outcome Numeric)"]
    Ytrain = Train.iloc[:, Train.columns == "CF (Outcome Numeric)"]

    LR = LogisticRegression(C=0.1, solver='newton-cg', max_iter=10000)
    LR.fit(Xtrain, Ytrain.values.ravel())
    #
    # 下面是这一次的score
    acc_score = LR.score(Xtest, Ytest)
    print("score", i, "\n", acc_score)
    sum.append(acc_score)
print("score mean\n", np.mean(sum))


print("\n\n\n\n以下是使用sklearn自带的方法进行划分和交叉验证")
# 划分训练集和测试集
data.drop(['Anon Student Id'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

#
# 重新整理了索引 这行代码写不写都行 可以不看
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
#
#
LR = LogisticRegression(C=0.1, solver='newton-cg',max_iter=10000)
LR.fit(Xtrain, Ytrain.values.ravel())
#
# 一次的score
acc_score = LR.score(Xtest, Ytest)
print("score\n", acc_score)


Ypred = LR.predict(Xtest)
print("AIC\n", aic(np.array(Ytest), Ypred, 12))
print("BIC\n", bic(np.array(Ytest), Ypred, 12))

#
# 进行交叉验证 十次
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(LR, X, y.values.ravel(), cv=10).mean()
print("score mean\n", cross_score)

