import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# 其余包在需要时引入，不统一写在顶部

data = pd.read_csv(r"/Users/LingZhang/Desktop/final_clean.csv")  # , index_col=0)
X = data.iloc[:, data.columns != "CF (Outcome Numeric)"]
y = data.iloc[:, data.columns == "CF (Outcome Numeric)"]

import sympy
import numpy as np


def get_mle(input_data):
    x, p, z = sympy.symbols('x p z', positive=True)
    phi = p ** x * (1 - p) ** (1 - x)
    L = np.prod([phi.subs(x, i) for i in input_data])
    logL = sympy.expand_log(sympy.log(L))
    sol, = sympy.solve(sympy.diff(logL, p), p)
    return sol

def get_bic(input_data):
    return (-2*get_mle(input_data)*input_data + k * np.log(input_data.shape[0]))