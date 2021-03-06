import pandas as pd
import numpy as np

# 其余包在需要时引入，不统一写在顶部


# # #
# STEP 1
# 导入.csv文件 路径如下
# 全部.csv文件都会存放在D:\Datasets中
print("################  STEP 1  ################")
data = pd.read_csv(r"/Users/song/Desktop/village_130.csv")  # , index_col=0)
print("##########################################\n\n\n\n")

# # #
# STEP 2
# 丢弃无用数据 并检查数据集
# 注意：有两个duration，我把有缺失值的那个删掉了
print("################  STEP 2  ################")
# data.drop(['Row'], axis=1, inplace=True)
data.drop(['Sample Name'], axis=1, inplace=True)
data.drop(['Transaction Id'], axis=1, inplace=True)
data.drop(['Time Zone'], axis=1, inplace=True)
data.drop(['Student Response Type'], axis=1, inplace=True)
data.drop(['Student Response Subtype'], axis=1, inplace=True)
data.drop(['Tutor Response Type'], axis=1, inplace=True)
data.drop(['Tutor Response Subtype'], axis=1, inplace=True)
data.drop(['Selection'], axis=1, inplace=True)
data.drop(['Action'], axis=1, inplace=True)
data.drop(['Feedback Text'], axis=1, inplace=True)
data.drop(['Feedback Classification'], axis=1, inplace=True)
data.drop(['Help Level'], axis=1, inplace=True)
data.drop(['Total Num Hints'], axis=1, inplace=True)
data.drop(['School'], axis=1, inplace=True)
data.drop(['Class'], axis=1, inplace=True)
data.drop(['CF (File)'], axis=1, inplace=True)
data.drop(['CF (Original DS Export File)'], axis=1, inplace=True)
data.drop(['CF (Village)'], axis=1, inplace=True)
data.drop(['CF (Week)'], axis=1, inplace=True)
data.drop(['CF (Total Activity Problems)'], axis=1, inplace=True)
data.drop(['CF (Session Sequence)'], axis=1, inplace=True)
data.drop(['CF (Unix Epoch)'], axis=1, inplace=True)
data.drop(['Session Id'], axis=1, inplace=True)
data.drop(['Time'], axis=1, inplace=True)
data.drop(['CF (Date)'], axis=1, inplace=True)
data.drop(['Step Name'], axis=1, inplace=True)
data.drop(['CF (Matrix Order)'], axis=1, inplace=True)
data.drop(['CF (Original Order)'], axis=1, inplace=True)
data.drop(['Problem Start Time'], axis=1, inplace=True)
data.drop(['CF (Activity Finished)'], axis=1, inplace=True)
data.drop(['CF (Activity Started)'], axis=1, inplace=True)
data.drop(['CF (Placement Test Flag)'], axis=1, inplace=True)
data.drop(['CF (Placement Test User)'], axis=1, inplace=True)
data.drop(['CF (Child Id)'], axis=1, inplace=True)
data.drop(['CF (Robotutor Mode)'], axis=1, inplace=True)
data.drop(['Problem View'], axis=1, inplace=True)
data.drop(['CF (Duration sec)'], axis=1, inplace=True)
data.drop(['Outcome'], axis=1, inplace=True)
# # #
data.info()  #

print("##########################################\n\n\n\n")

# # #
# STEP 3
# 缺失值处理
"""
排除掉肯定不用的column之后检查数据集，
①Input 两个缺失值，丢弃
②CF(Expected Answer) 几十个缺失值，丢弃
③CF(Hiatus sec) 几十个缺失值，丢弃
④CF (Student Used Scaffold) 大量缺失值，0填充
⑥此外还要把所有的object转为数字
"""
print("################  STEP 3  ################")
print("对于缺失特别多的scaffold，对其进行缺失值填补")
# 缺失值填补 CF (Student Used Scaffold)
label1 = data['CF (Student Used Scaffold)'].unique().tolist()
# 转数字
data['CF (Student Used Scaffold)'] = data['CF (Student Used Scaffold)'].apply(lambda x: label1.index(x))
temp = data.loc[:, "CF (Student Used Scaffold)"].values.reshape(-1, 1)  # sklearn当中特征矩阵必须是二维
from sklearn.impute import SimpleImputer

imp_0 = SimpleImputer(strategy="constant", fill_value=0)  # 用0填补
imp_0 = imp_0.fit_transform(temp)

# 在这里用0填补
data.loc[:, "CF (Student Used Scaffold)"] = imp_0

# 填补完之后 考虑到剩下的column缺失值很少而且都是难以填补的 直接丢弃
data = data.dropna()
data.info()  # 检查
print("##########################################\n\n\n\n")

# # #
# STEP 4
# 把剩下的object类型悉数转换为数字
print("################  STEP 4  ################")
# 先把最好处理的搞掉
# Matrix
label2 = data['CF (Matrix)'].unique().tolist()
data['CF (Matrix)'] = data['CF (Matrix)'].apply(lambda x: label2.index(x))

# Input
label3 = data['Input'].unique().tolist()
data['Input'] = data['Input'].apply(lambda x: label3.index(x))

# Expected Answer
label4 = data['CF (Expected Answer)'].unique().tolist()
data['CF (Expected Answer)'] = data['CF (Expected Answer)'].apply(lambda x: label4.index(x))
#
# data = pd.get_dummies(data, columns=['Anon Student Id'])

# Anon Student Id
label5 = data['Anon Student Id'].unique().tolist()
data['Anon Student Id'] = data['Anon Student Id'].apply(lambda x: label5.index(x))

# Tutor Name
label6 = data["Level (Tutor Name)"].unique().tolist()
data["Level (Tutor Name)"] = data["Level (Tutor Name)"].apply(lambda x: label6.index(x))

# Tutor
text1 = data["Level (Tutor)"]
# 截取string字符串实现分类
d_list = text1.str.split(".").tolist()
print(d_list)
# 获取分类列表，实现去重
b_list = [n for m in d_list for n in m]

category_column = list(set(b_list))
data_tutor = pd.DataFrame(np.zeros((data.shape[0], len(category_column))), columns=category_column)

# 下面注释掉的代码可以用来生成一个Level(Tutor)的词矩阵
"""
print("#### Level (Tutor) DataFrame 生成中 ####")
for m in range(data.shape[0]):
    # print(m, d_list[m])
    data_tutor.loc[m, d_list[m]] = 1


data_tutor['Row'] = data['Row']

data_tutor.to_csv(r'D:\Datasets\village_130_tutor.csv', sep=',', header=True, index=False)
print('#### Level (Tutor) DataFrame 已生成 ####')
"""
data.drop(['Level (Tutor)'], axis=1, inplace=True)

# Problem Name
label7 = data["Problem Name"].unique().tolist()
data["Problem Name"] = data["Problem Name"].apply(lambda x: label7.index(x))
# print(data_problem.shape)

print('data_tutor  ', data_tutor.shape)
print('data  ', data.shape)

# 处理duration，字符串直接转数字
data["Duration (sec)"] = pd.to_numeric(data["Duration (sec)"], errors='coerce')
# print(data['Duration (sec)'])
data = data.dropna()
data.info()
print("##########################################\n\n\n\n")

# """
# # #
# STEP 5
# 归一化
print("################  STEP 5  ################")
# 需要归一化处理的column
column_list = ['Duration (sec)',
               # 'Attempt At Step'
               # 'CF (Attempt Number)',
               'CF (Hiatus sec)',
               # 'CF (Matrix Level)'
               ]

# 下面代码存在问题，会删除全部数据
# print(data)
backupdata = data[['CF (Hiatus sec)']]
print(backupdata)
from scipy import stats

constrains = backupdata.apply(lambda x: np.abs(stats.zscore(x)) < 2).all(axis=1)
# Drop (inplace) values set to be rejected
data.drop(backupdata.index[~constrains], inplace=True)
print('print', data)

# 下面的代码可以用来删除过长的Duration
data = data.loc[(data['Duration (sec)'] < 300 )& (data['Duration (sec)']  > 0)]
# while data['Duration (sec)'].idxmax() > 300.0:
#     data = data.drop(data['Duration (sec)'].idxmax())
# while data['Duration (sec)'].idxmin() <= 0.0:
#     data = data.drop(data['Duration (sec)'].idxmin())

#
for i in column_list:
    Max = np.max(data[i])
    Min = np.min(data[i])
    print(i, ' ', "Max = ", Max, "Min = ", Min)
    data['Regularization ' + i] = (data[i] - Min) / (Max - Min)

#
# data.info()
print(data[['Duration (sec)', 'Regularization Duration (sec)']])

print("##########################################\n\n\n\n")

# 将预处理完成后的数据保存的village_130_1.csv中
data.to_csv(r'/Users/song/Desktop/village_130_1.csv', sep=',', header=True, index=False)
