import pandas as pd
import numpy as np
import random

data = pd.read_csv(r"D:\Datasets\village_130_1.csv")#, index_col=0)
data['Outcome'] = data['CF (Outcome Numeric)'] #把outcome移动到最后一列
data.drop(['CF (Outcome Numeric)'], axis=1, inplace=True)

data.drop(['Row'], axis=1, inplace=True)
# data.drop(['Is Last Attempt'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
# data.drop(['CF (Attempt Number)'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['CF (Hiatus sec)'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
data.drop(['CF (Student Chose Repeat)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
# data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Anon Student Id'], axis=1, inplace=True)
data.drop(['Duration (sec)'], axis=1, inplace=True)
data.info()
# print(data)

l = list(data.index) # 得到索引
random.shuffle(l) # 随机排序
data.index = l # 随机排序后的索引赋给数据集
n = data.shape[0] # 行数
m = int(n * 0.8) # 训练集数量
train = data.loc[range(m), :] # 前m个训练集
test = data.loc[range(m, n), :] #测试集

# 更新一下索引
data.index = range(data.shape[0])
test.index = range(test.shape[0])

# print(train)
# print(test)
# print(l)


labels = train.iloc[:, -1].value_counts().index
mean = []
std = []
result = []
#print(labels[0])

for i in labels:
    item = train.loc[train.iloc[:, -1] == i, :]
    m = item.iloc[:, :-1].mean()
    s = np.sum((item.iloc[:, :-1]-m)**2)/(item.shape[0])
    mean.append(m)
    std.append(s)
means = pd.DataFrame(mean, index=labels)
stds = pd.DataFrame(std, index=labels)
# print(mean)
# print(std)

for j in range(test.shape[0]):
    iset = test.iloc[j, :-1].tolist()
    iprob = np.exp(-1*(iset-means)**2/(stds*2))/(np.sqrt(2*np.pi*stds))
    prob = 1
    for k in range(test.shape[1]-1):
        prob *= iprob.iloc[:, k]
        cla = prob.index[np.argmax(prob.values)]
    result.append(cla)
test['Predict'] = result
acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()
print('计算中')
print(acc)
