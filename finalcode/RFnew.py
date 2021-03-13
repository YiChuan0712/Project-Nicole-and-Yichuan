# 随机森林
# 下面是一部分导入的包 剩下的包在使用时才导入（尽管这并不是一个良好的代码习惯）
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
# data = pd.read_csv(r"/Users/LingZhang/Desktop/final_clean.csv")  # , index_col=0)
data = pd.read_csv(r"D:\Datasets\final_clean.csv")  # , index_col=0)
data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
##################

#
data.drop(['CF (Matrix)_stories'], axis=1, inplace=True)
#
#
#

# data.info()
X = data.iloc[:, data.columns != "CF (Outcome Numeric)"]
y = data.iloc[:, data.columns == "CF (Outcome Numeric)"]
#
#
#


list = data['Anon Student Id'].unique()
# print(list)
#
print("\n\n\n\n逻辑回归")
print("\n\n\n\n以下是每次踢出一个学生的交叉验证方法")
sum = []
from sklearn.metrics import roc_curve,auc
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)
for i in [0,2,3,4,5,6,7,8]:
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

    RF = RandomForestClassifier(criterion='gini', max_depth=2, min_impurity_decrease=0.0, min_samples_leaf=1)
    RF.fit(Xtrain, Ytrain.values.ravel())
    Ypred = RF.predict(Xtest)

    RF.fit(Xtrain, Ytrain.values.ravel())
    target_probabilities = RF.predict_proba(Xtest)[:, 1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(Ytest, target_probabilities)

    from sklearn.metrics import roc_curve, roc_auc_score

    # interp 插值
    tprs.append(np.interp(mean_fpr, false_positive_rate, true_positive_rate))
    tprs[-1][0] = 0.0
    # auc
    roc_auc = auc(false_positive_rate, true_positive_rate)
    aucs.append(roc_auc)

    # 下面的代码别动
    if i == 0:
        i = 1
    i = i-1
    #
    plt.plot(false_positive_rate, true_positive_rate, lw=1, alpha=0.5, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
# print(std_tpr)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()

"""
# 可以算的比较多 我统一写在注释里 以备不时之需

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("TP/(TP+FN)    True Positive Rate/Sensitivity/Hit Rate/Recall\n", TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP)
print("TN/(TN+FP)    True Negative Rate/Specificity\n", TNR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print("TP/(TP+FP)    Positive Predictive Value/Precision\n", PPV)

# Negative predictive value
NPV = TN/(TN+FN)
print("TN/(TN+FN)    Negative Predictive Value\n", NPV)

# Fall out or false positive rate
FPR = FP/(FP+TN)
print("FP/(FP+TN)    False Positive Rate/Fall Out\n", FPR)

# False negative rate
FNR = FN/(TP+FN)
print("FN/(TP+FN)    False Negative Rate\n", FNR)

# False discovery rate
FDR = FP/(TP+FP)
print("FP/(TP+FP)    False Discovery Rate\n", FDR)


# precision = TP / (TP+FP)  # 查准率
# recall = TP / (TP+FN)  # 查全率
"""

"""
best params
 {'criterion': 'gini', 'max_depth': 2, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1}
best score
 0.8550277468421903
"""

"""
tr = []
te = []
for i in range(10):
    rf = RandomForestClassifier(random_state=25, max_depth=i+1)
    rf = rf.fit(Xtrain, Ytrain)
    score_tr = rf.score(Xtrain, Ytrain)
    score_te = cross_val_score(rf, X, y.values.ravel(), cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1,11), tr, color='red', label='train')
plt.plot(range(1,11), te, color='blue', label='test')
plt.xticks(range(1,11))
plt.show()
"""

"""
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")

import numpy as np

parameters = {'criterion': ("gini", "entropy")
                , "max_depth": [*range(2, 4)]
                , 'min_samples_leaf': [*range(1, 50, 5)]
                , 'min_impurity_decrease': [*np.linspace(0, 0.5, 20)]
              }

lgt = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
GS = GridSearchCV(lgt, parameters, cv=10)
GS = GS.fit(Xtrain, Ytrain.values.ravel())

print("best params\n", GS.best_params_)

print("best score\n", GS.best_score_)
"""
