"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"/Users/song/Desktop/village_130_1.csv")  # , index_col=0)
data.info()

pd.set_option('display.max_columns', 20)
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
"""
# 逻辑回归 修改版
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
# 导入.csv文件
data = pd.read_csv(r"D:\Datasets\village_130_1.csv")  # , index_col=0)
#
#
#
# 画图（完全保留了上一版本的代码，未作任何修改）
pd.set_option('display.max_columns', 5)
pd.set_option('display.expand_frame_repr', False)
corr = data[['Duration (sec)', 'CF (Matrix)', 'Attempt At Step', 'Is Last Attempt', 'CF (Attempt Number)',
             'CF (Student Chose Repeat)', 'CF (Student Used Scaffold)']].corr()
print(corr)
heat_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()
#
#
#
# drop掉的column 我个人建议使用这种写法
# 因为现阶段还没有完全确定下来predictor 这样写便于逐个column进行调整 想保留column 注释掉对应行即可
#
# data.drop(['Duration (sec)'], axis=1, inplace=True)
# data.drop(['Attempt At Step'], axis=1, inplace=True)
# data.drop(['Is Last Attempt'], axis=1, inplace=True)
# data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
# data.drop(['CF (Matrix)'], axis=1, inplace=True)
# data.drop(['CF (Outcome Numeric)'], axis=1, inplace=True)
# data.drop(['CF (Student Chose Repeat)'], axis=1, inplace=True)
# data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['Row'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['CF (Hiatus sec)'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
data.drop(['Level (Tutor Name)'], axis=1, inplace=True)
data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Anon Student Id'], axis=1, inplace=True)
data.drop(['CF (Matrix Level)'], axis=1, inplace=True)
data.info()# 检查
#
#
#
#
# 划分Xy 注意X是矩阵 y是向量 建议X大写 y小写
X = data.iloc[:, data.columns != "CF (Outcome Numeric)"]
y = data.iloc[:, data.columns == "CF (Outcome Numeric)"]
#
#
#
# 划分训练集和测试集 !!!注意!!! 我修改了原来的代码
# “把每个学生数据按0.8和0.2的训练数据和测试数据，然后再拼到一起” 这个操作是没有意义的
# 因为 train_test_split 本身就是随机拆分
# 把大数据集分成几份分别随机二八分在合并 这和直接二八分是没有区别的
# 因此 这里直接拆分train test
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

#
# 重新整理了索引 这行代码写不写都行 可以不看
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
#
#
logit = LogisticRegression(C=0.4, solver='lbfgs',max_iter=10000)
logit.fit(Xtrain, Ytrain.values.ravel())
#
# 下面是这一次的score
acc_score = logit.score(Xtest, Ytest)
print("\nscore")
print(acc_score)
#
# 进行交叉验证 十次
from sklearn.model_selection import cross_val_score
cross_score = cross_val_score(logit, X, y.values.ravel(), cv=10).mean()
print("\nscore mean")
print(cross_score)
#
#
#
# 至此 完成了粗跑 也就是说LogisticRegression()暂时没有填写任何参数
# 但效果还可以 因为两个score都在90左右 说明没有出现严重的过拟合 模型的表现也很稳定
#
# 接下来要利用图像和网格搜索 寻找最佳的参数 并回填到LogisticRegression()中
# 首先查资料 看LogisticRegression()的重要参数有哪些
# 阅读资料后我保留了以下几个参数进行研究
#
# penalty 用来选择正则化方式 不填默认'l2' 这里不填即可
# C 正则化强度的倒数 越小惩罚越重 默认1.0
# max_iter 梯度下降的最大迭代次数 搞长点应该没什么坏处
# solver 求解器 一共五种 我也不会，所以让网格搜索帮我找
# 五个分别是 liblinear lbfgs newton-cg sag saga
# multi_class 默认值是auto 不用管
#
# 简单的网格搜索如下 注意 非常耗时 我已经将计算得到的结果进行记录
# best params
#  {'C': 0.4, 'solver': 'lbfgs'}
# best score
#  0.9123283582089552
"""
print("\nTIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")

parameters = {'solver': ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga')
                , 'C': [*np.linspace(0.1, 2.0, 20)]
              }

lgt = LogisticRegression()
from sklearn.model_selection import GridSearchCV
GS = GridSearchCV(lgt, parameters, cv=10)
GS = GS.fit(Xtrain, Ytrain.values.ravel())

print("best params\n", GS.best_params_)

print("best score\n", GS.best_score_)
# """
