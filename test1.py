import pandas as pd
import numpy as np

# # #
# STEP 1
# 导入.csv文件 路径如下
print("################  STEP 1  ################")
data = pd.read_csv(r"D:\Datasets\village_130.csv")#, index_col=0)
print("##########################################\n\n\n\n")




# # #
# STEP 2
# 丢弃无用数据 并检查数据集
# 注意：有两个duration，我把有缺失值的那个删掉了
print("################  STEP 2  ################")
data.drop(['Row'], axis=1, inplace=True)
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
# # #
data.drop(['CF (Village)'], axis=1, inplace=True)
data.drop(['CF (Week)'], axis=1, inplace=True)
data.drop(['CF (Total Activity Problems)'], axis=1, inplace=True)
data.drop(['CF (Session Sequence)'], axis=1, inplace=True)
data.drop(['CF (Unix Epoch)'], axis=1, inplace=True)
# # #
data.drop(['Session Id'], axis=1, inplace=True)
data.drop(['Time'], axis=1, inplace=True)
data.drop(['CF (Date)'], axis=1, inplace=True)
data.drop(['Step Name'], axis=1, inplace=True)
data.drop(['CF (Matrix Order)'], axis=1, inplace=True)
data.drop(['CF (Original Order)'], axis=1, inplace=True)
data.drop(['Problem Start Time'], axis=1, inplace=True)
data.drop(['CF (Activity Finished)'], axis=1, inplace=True)
data.drop(['CF (Activity Started)'], axis=1, inplace=True)
# # #
data.drop(['CF (Placement Test Flag)'], axis=1, inplace=True)
data.drop(['CF (Placement Test User)'], axis=1, inplace=True)
data.drop(['CF (Child Id)'], axis=1, inplace=True)
data.drop(['CF (Robotutor Mode)'], axis=1, inplace=True)
data.drop(['Problem View'], axis=1, inplace=True)
data.drop(['CF (Duration sec)'], axis=1, inplace=True)
data.drop(['Outcome'], axis=1, inplace=True)
# # #
data.info()#
print('排除掉肯定不用的column之后检查数据集，剩下问题的主要集中在下面几列')

print('①Input 两个缺失值')

print('②CF(Expected Answer) 几十个缺失值')

print('③CF(Hiatus sec) 几十个缺失值')

print('④CF (Student Used Scaffold) 大量缺失值')

print('⑤上面几个缺失很少可以直接删除 但是Scaffold不行，必须对缺失值进行填补')

print('⑥此外还要把所有的object转为数字')

print("##########################################\n\n\n\n")




# # #
# STEP 3
# 缺失值处理
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

data.info()# 检查
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


# data=pd.DataFrame(data, dtype=np.float) #这条语句用来处理duration

data.info()
print("##########################################\n\n\n\n")
"""

Time spent on similar previous problems-duration

Number of attempting

Historical % correct on "similar" items ( column: outcome)

whether student used scaffold: Hint / Not hint / Quit / NAN
whether a student used scaffold on similar items
CF student chooses to repeat #
Matrix Level, the difficulty of problems
Whether the student chooses to repeat.
Tutor Sequence User, the number of tutors a student experienced, may affect the Outcome (also Tutor Sequence Session）.

"""