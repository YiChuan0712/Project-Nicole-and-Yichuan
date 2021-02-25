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
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
data.info()
corr = data[['CF (Duration sec)', 'CF (Matrix Level)', 'Level (Tutor Name)_akira', 'Level (Tutor Name)_bigmath', 'Level (Tutor Name)_bubble_pop', 'Level (Tutor Name)_picmatch',
             'Level (Tutor Name)_spelling', 'Level (Tutor Name)_story_questions', 'Level (Tutor Name)_story_reading', 'Level (Tutor Name)_word_copy',
             'CF (Matrix)_literacy', 'CF (Matrix)_math', 'CF (Matrix)_stories']].corr()
print(corr)
heat_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
#
data.drop(['CF (Matrix)_stories'], axis=1, inplace=True)
#
#
#
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
data.info()
corr = data[['CF (Duration sec)', 'CF (Matrix Level)', 'Level (Tutor Name)_akira', 'Level (Tutor Name)_bigmath', 'Level (Tutor Name)_bubble_pop' , 'Level (Tutor Name)_picmatch',
             'Level (Tutor Name)_spelling','Level (Tutor Name)_story_questions', 'Level (Tutor Name)_story_reading', 'Level (Tutor Name)_word_copy',
             'CF (Matrix)_literacy', 'CF (Matrix)_math']].corr()
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
print("\n\n\n\n逻辑回归")
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

    RF = RandomForestClassifier(criterion='gini', max_depth=2, min_impurity_decrease=0.0, min_samples_leaf=1)
    RF.fit(Xtrain, Ytrain.values.ravel())
    #
    # 下面是这一次的score
    acc_score = RF.score(Xtest, Ytest)
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
RF = RandomForestClassifier(criterion='gini', max_depth=2, min_impurity_decrease=0.0, min_samples_leaf=1)
RF.fit(Xtrain, Ytrain.values.ravel())
#
# 一次的score
acc_score = RF.score(Xtest, Ytest)
print("score\n", acc_score)


Ypred = RF.predict(Xtest)
print("AIC\n", aic(np.array(Ytest), Ypred, 12))
print("BIC\n", bic(np.array(Ytest), Ypred, 12))

#
# 进行交叉验证 十次
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(RF, X, y.values.ravel(), cv=10).mean()
print("score mean\n", cross_score)


print("\n\n\n\nROC和AUC")
from sklearn.metrics import roc_curve, roc_auc_score
RF = RandomForestClassifier(criterion='gini', max_depth=2, min_impurity_decrease=0.0, min_samples_leaf=1)
RF.fit(Xtrain, Ytrain.values.ravel())
target_probabilities = RF.predict_proba(Xtest)[:, 1]
false_positive_rate, true_positive_rate, threshold = roc_curve(Ytest, target_probabilities)
# print("false positive rate\n", true_positive_rate)
# print("true positive rate\n", false_positive_rate)

# 图像标题
plt.title("ROC Curve - RF Model")
# 横轴纵轴数据
plt.plot(false_positive_rate, true_positive_rate, label='LR')
plt.legend(loc='best')
plt.plot([0, 1], ls="--", color='black')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
print('AUC\n', roc_auc_score(Ytest, target_probabilities))


print("\n\n\n\n混淆矩阵")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


Ypred = RF.predict(Xtest)

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
