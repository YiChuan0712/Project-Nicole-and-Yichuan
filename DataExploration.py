import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"/Users/LingZhang/Desktop/village_130.csv")
# data.info()
data['Outcome'] = data['CF (Outcome Numeric)']
data.drop(['CF (Outcome Numeric)'], axis=1, inplace=True)
# DROP
data.drop(['Row'], axis=1, inplace=True)
data.drop(['Is Last Attempt'], axis=1, inplace=True)
data.drop(['Input'], axis=1, inplace=True)
data.drop(['CF (Expected Answer)'], axis=1, inplace=True)
data.drop(['CF (Hiatus sec)'], axis=1, inplace=True)
data.drop(['CF (Problem Number)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence Session)'], axis=1, inplace=True)
data.drop(['CF (Student Used Scaffold)'], axis=1, inplace=True)
data.drop(['CF (Tutor Sequence User)'], axis=1, inplace=True)
# data.drop(['Problem Name'], axis=1, inplace=True)
data.drop(['Anon Student Id'], axis=1, inplace=True)
data.drop(['Duration (sec)'], axis=1, inplace=True)
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
data.drop(['CF (Placement Test Flag)'], axis=1, inplace=True)
data.drop(['CF (Placement Test User)'], axis=1, inplace=True)
data.drop(['CF (Village)'], axis=1, inplace=True)
data.drop(['CF (Original DS Export File)'], axis=1, inplace=True)
data.drop(['CF (Robotutor Mode)'], axis=1, inplace=True)
data.drop(['CF (Week)'], axis=1, inplace=True)
data.drop(['CF (Session Sequence)'], axis=1, inplace=True)


data.info()


# 1.b correct number -> correct rate
# related_df = data[
#     ['Duration (sec)', 'CF (Attempt Number)', 'Input', 'Level (Tutor Name)', 'CF (Outcome Numeric)', 'CF (Matrix)',
#      'CF (Matrix Level)', 'Anon Student Id', 'CF (Tutor Sequence User)', 'CF (Student Chose Repeat)', 'Is Last Attempt',
#      'CF (Student Used Scaffold)', 'Attempt At Step', 'CF (Placement Test User)']];
#
#
# def get_correct_rate_chart(source_df, column_name):
#     # df = source_df.loc[related_df['CF_Attempt_Number'] == 1]
#     df = source_df.groupby([column_name])['CF (Outcome Numeric)'].agg({'count', 'sum'}).reset_index()
#     df['correct_rate'] = df['sum'] / df['count']
#     df = df.reset_index()
#     sns.barplot(y='correct_rate', data=df, x=column_name)
#     plt.show()
#
#
# def get_count_chart(source_df, column_name):
#     df = source_df.groupby([column_name, 'CF (Outcome Numeric)'])['Duration (sec)'].count().reset_index()
#     df.columns = [column_name, 'CF (Outcome Numeric)', 'count']
#     sns.barplot(hue=column_name, y='count', data=df, x='CF (Outcome Numeric)')
#     plt.show()
#
# #Analysting student's first attempt
#
# # get_correct_rate_chart(related_df, 'CF (Matrix Level)')
#
# get_count_chart(related_df, 'CF (Matrix)')
#
# get_count_chart(related_df, 'Level (Tutor Name)')

# get_correct_rate_chart(related_df, 'CF (Placement Test User)')
#
# get_correct_rate_chart(related_df, 'CF (Student Chose Repeat)')
#
# get_correct_rate_chart(related_df, 'Is Last Attempt')
#
# get_correct_rate_chart(related_df, 'CF (Student_Used_Scaffold)')
