import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Datasets\tutorname_duration.csv")

feature_names = ['akira', 'bigmath', 'bubble_pop', 'picmatch', 'spelling', 'story_questions', 'story_reading', 'word_copy']

feature_data = {}
for f in feature_names:
    feature_data[f] = df[f]

data = pd.DataFrame(feature_data)
data.plot.box(title="test figure")
plt.ylim(0, 110)
plt.show()
