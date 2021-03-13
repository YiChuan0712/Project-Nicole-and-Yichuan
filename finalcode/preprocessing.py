import pandas as pd
import numpy as np
# 其余包在需要时引入，不统一写在顶部
#
#
#
#
# # #
# STEP 1
# 导入.csv文件 路径如下
print("################  STEP 1  ################")
print('################  导入.csv文件')
data = pd.read_csv(r"/Users/song/Desktop/clean.csv")  # , index_col=0)
# data = pd.read_csv(r"D:\Datasets\clean.csv")#, index_col=0)
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 2
# 丢弃一部分无用数据 并检查数据集
print("################  STEP 2  ################")
print('################  drop一部分无用数据')
# data.info()
data.drop(['Row'], axis=1, inplace=True)
data.drop(['CF (File)'], axis=1, inplace=True)
data.drop(['CF (Week)'], axis=1, inplace=True)
data.drop(['CF (Total Activity Problems)'], axis=1, inplace=True)
data.drop(['CF (Session Sequence)'], axis=1, inplace=True)
data.drop(['Session Id'], axis=1, inplace=True)
data.drop(['CF (Date)'], axis=1, inplace=True)
data.drop(['CF (Matrix Order)'], axis=1, inplace=True)
data.drop(['CF (Original Order)'], axis=1, inplace=True)
data.drop(['CF (Activity Finished)'], axis=1, inplace=True)
data.drop(['CF (Activity Started)'], axis=1, inplace=True)
data.drop(['CF (Child Id)'], axis=1, inplace=True)
data.drop(['CF (Robotutor Mode)'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['Outcome'], axis=1, inplace=True)
# data.drop(['Anon Student Id'], axis=1, inplace=True)
# data.drop(['Level (Tutor)'], axis=1, inplace=True)
data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Is Last Attempt'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
data.drop(['CF (Student Chose Repeat)'], axis=1, inplace=True)
data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
# # #
data.info()  #
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 3
# 删除countingx
print("################  STEP 3  ################")
print('################  删除countingx')
# 删除countingx 因为countingx的outcome必定为正确
data = data[~data['Level (Tutor Name)'].isin(['countingx'])]
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 4
# 删除story_hear
print("################  STEP 4  ################")
print('################  删除story.hear')
# 删除story_hear
# 注意在step 1中我并没有删去level(tutor) 就是为了在这里使用
# 首先选出含有‘story hear’的level tutor
_data = data[data['Level (Tutor)'].str.contains('story.hear')]
# _data.info()
# data.info()
test1 = list(_data['Level (Tutor)'])
# print(test1)
test2 = list(data['Level (Tutor)'])
# 求差集 得到所有不含story hear的
ret = list(set(test2) ^ set(test1))
data = data[data['Level (Tutor)'].isin(ret)]
# 做完这一步之后就可以删除level tutor
data.drop(['Level (Tutor)'], axis=1, inplace=True)
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 5
# 删除所有为0的duration
print("################  STEP 5  ################")
print('################  删除0 duration')
# 删除duration=0的行 取对数方便
# print(data[data['CF (Duration sec)'].isin([0])]['CF (Duration sec)'])
data = data[~data['CF (Duration sec)'].isin([0])]
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 6
# 处理object类型
print("################  STEP 6  ################")
print('################  object转int')
# data.info()
# Anon Student Id 转化为数字 注意 在进行预测时要去掉这一列
# 这里转化成数字 只是为了在进行交叉验证时方便
label1 = data['Anon Student Id'].unique().tolist()
data['Anon Student Id'] = data['Anon Student Id'].apply(lambda x: label1.index(x))
# print(data['Anon Student Id'].unique())
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 7
# 处理缺失值和object类型
print("################  STEP 7  ################")
print('################  删除含有缺失值的行')
data = data.dropna()
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 8
# 处理duration
# 先取对数再/log(max)
# 然后超过一定标准差(3个，按照tutorname分类)就删除
print("################  STEP 8  ################")
print('################  Normalization')
#
#
import matplotlib.pyplot as plt
import seaborn as sns
#
# 数据集右偏严重 所以取个对数
# 可以画图来看
#
# 以全部的duration
# 取对数前
# sns.distplot(data['CF (Duration sec)'])
# # 取对数后
# sns.distplot(data['CF (Duration sec)'].apply(np.log))
#
# 以spelling的duration为例
# temp = data[data['Level (Tutor Name)'].isin(['spelling'])]
# 前
# sns.distplot(temp['CF (Duration sec)'])
# 后
# sns.distplot(temp['CF (Duration sec)'].apply(np.log))
#
# plt.show()
#
#
column_list = ['CF (Duration sec)']
#
for i in column_list:
    Max = np.max(data[i])
    Min = np.min(data[i])
    print(i, ' ', "Max = ", Max, "Min = ", Min)
    # data[i] = (data[i] - Min)/(Max - Min)

# sns.distplot(data['CF (Duration sec)'])
#
# sns.distplot(data['CF (Duration sec)'].apply(np.log)/np.log(Max))
#
# plt.show()

# 取对数/np.log(Max)
data['CF (Duration sec)'] = data['CF (Duration sec)'].apply(np.log)/np.log(Max)
# data.to_csv(r"D:\Datasets\mid_clean.csv", sep=',', header=True, index=False)
# 列出标准差
_std = data.groupby('Level (Tutor Name)')[['CF (Duration sec)']].std()
# _median = data.groupby('Level (Tutor Name)')[['CF (Duration sec)']].median()
print(_std)
# print(_median)
# 3个标准差
n = 3
datamean = data[(data['Level (Tutor Name)'] == 'akira')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'akira') & (data['CF (Duration sec)'] - datamean > n * 0.033961)].index)
datamean = data[(data['Level (Tutor Name)'] == 'bigmath')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'bigmath') & (data['CF (Duration sec)'] - datamean > n * 0.081707)].index)
datamean = data[(data['Level (Tutor Name)'] == 'bubble_pop')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'bubble_pop') & (data['CF (Duration sec)'] - datamean > n * 0.074145)].index)
datamean = data[(data['Level (Tutor Name)'] == 'picmatch')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'picmatch') & (data['CF (Duration sec)'] - datamean > n * 0.424925)].index)
datamean = data[(data['Level (Tutor Name)'] == 'spelling')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'spelling') & (data['CF (Duration sec)'] - datamean > n * 0.152053)].index)
datamean = data[(data['Level (Tutor Name)'] == 'story_questions')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'story_questions') & (data['CF (Duration sec)'] - datamean > n * 0.173010)].index)
datamean = data[(data['Level (Tutor Name)'] == 'story_reading')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'story_reading') & (data['CF (Duration sec)'] - datamean > n * 0.127753)].index)
datamean = data[(data['Level (Tutor Name)'] == 'word_copy')]['CF (Duration sec)'].mean()
data = data.drop(data[(data['Level (Tutor Name)'] == 'word_copy') & (data['CF (Duration sec)'] - datamean > n * 0.109107)].index)
#
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 9
# 独热编码
print("################  STEP 9  ################")
print('################  One Hot')
# 独热编码 get dummies
data = pd.get_dummies(data)
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 10
# 2.23紧急更新 保留attempt number == 1
print("################  STEP 10  ################")
print('################  保留 attempt number 1')
data = data[data['CF (Attempt Number)'].isin([1])]
data.info()
print("##########################################\n\n\n\n")
#
#
#
#
# # #
# STEP 11
# 将预处理完成后的数据保存到final_clean.csv中
print("################  STEP 11  ################")
print('################  保存数据')
data.to_csv(r'/Users/song/Desktop/final_clean.csv', sep=',', header=True, index=False)
# data.to_csv(r"D:\Datasets\final_clean.csv", sep=',', header=True, index=False)
print("##########################################\n\n\n\n")

