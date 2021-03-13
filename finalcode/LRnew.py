# 逻辑回归
# 下面是一部分导入的包 剩下的包在使用时才导入（尽管这并不是一个良好的代码习惯）
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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

data.info()
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
aics=[]
bics=[]
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

    LR = LogisticRegression(max_iter=10000, C=0.1, penalty='l2', solver='newton-cg')
    LR.fit(Xtrain, Ytrain.values.ravel())

    Ypred = LR.predict(Xtest)
    #print(aic(np.array(Ytest), Ypred, 12))
    #bics.append(bic(np.array(Ytest), Ypred, 12))

    LR.fit(Xtrain, Ytrain.values.ravel())
    target_probabilities = LR.predict_proba(Xtest)[:, 1]
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


print(np.mean(aucs))
print(np.mean(aics))
print(np.mean(bics))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)


std_tpr = np.std(tprs, axis=0)
# print(std_tpr)
tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
# plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
#plt.xlim([-0.01, 0.1])
#plt.ylim([-0.05, 0.65])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()




"""
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
tns = []
fps = []
fns = []
tps = []
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

    LR = LogisticRegression(max_iter=10000, C=0.1, penalty='l2', solver='newton-cg')
    LR.fit(Xtrain, Ytrain.values.ravel())

    Ypred = LR.predict(Xtest)

    LR.fit(Xtrain, Ytrain.values.ravel())

    TN, FP, FN, TP = confusion_matrix(Ytest, Ypred).ravel()
    tns.append(TN)
    fps.append(FP)
    fns.append(FN)
    tps.append(TP)

TN = np.mean(tns)
FP = np.mean(fps)
FN = np.mean(fns)
TP = np.mean(tps)

SUM = TN + FP + FN + TP
 
confusion = [[TP, FN], [FP, TN]]
print(confusion)

plt.imshow(confusion, cmap=plt.cm.Reds)

indices = range(len(confusion))

plt.xticks(indices, ['Correct', 'Incorrect'])
plt.yticks(indices, ['Correct', 'Incorrect'])

plt.colorbar()

plt.xlabel('Predict')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.text(0, 0, confusion[0][0])
plt.text(1, 0, confusion[0][1])
plt.text(0, 1, confusion[1][0])
plt.text(1, 1, confusion[1][1])

plt.show()

print('TP\n', TP)
print('TN\n', TN)
print('FP\n', FP)
print('FN\n', FN)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("TP/(TP+FN)    Sensitivity\n", TPR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print("TP/(TP+FP)    Precision\n", PPV)
"""

print("\n\n\n\n混淆矩阵")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

LR = LogisticRegression(max_iter=10000, C=0.1, penalty='l2', solver='newton-cg')
LR.fit(Xtrain, Ytrain.values.ravel())

Ypred = LR.predict(Xtest)

TN, FP, FN, TP = confusion_matrix(Ytest, Ypred).ravel()

confusion = [[TP, FN], [FP, TN]]
print(confusion)

plt.imshow(confusion, cmap=plt.cm.Reds)

indices = range(len(confusion))

plt.xticks(indices, ['Correct', 'Incorrect'])
plt.yticks(indices, ['Correct', 'Incorrect'])

plt.colorbar()

plt.xlabel('Predict')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.text(0, 0, confusion[0][0])
plt.text(1, 0, confusion[0][1])
plt.text(0, 1, confusion[1][0])
plt.text(1, 1, confusion[1][1])

plt.show()

print('TP\n', TP)
print('TN\n', TN)
print('FP\n', FP)
print('FN\n', FN)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print("TP/(TP+FN)    Sensitivity\n", TPR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print("TP/(TP+FP)    Precision\n", PPV)



"""
best params
 {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
best score
 0.8549506458398775
"""
"""
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")

import numpy as np

parameters = {'penalty': ("l1", "l2"),
'solver': ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga')
                , 'C': [*np.linspace(0.1, 10, 20)]
              }

lgt = LogisticRegression()
from sklearn.model_selection import GridSearchCV
GS = GridSearchCV(lgt, parameters, cv=10)
GS = GS.fit(Xtrain, Ytrain.values.ravel())

print("best params\n", GS.best_params_)

print("best score\n", GS.best_score_)
"""
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
