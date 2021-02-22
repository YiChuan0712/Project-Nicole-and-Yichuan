import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"/Users/LingZhang/Desktop/final_clean.csv")  # , index_col=0)

sns.distplot(data['CF (Duration sec)'])
plt.show()