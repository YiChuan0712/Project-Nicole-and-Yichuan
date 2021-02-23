import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# 其余包在需要时引入，不统一写在顶部

data = pd.read_csv(r"/Users/LingZhang/Desktop/final_clean.csv")  # , index_col=0)
data.drop(['CF (Matrix)_stories'], axis=1, inplace=True)
data.drop(['CF (Attempt Number)'], axis=1, inplace=True)

X = data.iloc[:, data.columns != "CF (Outcome Numeric)"]
y = data.iloc[:, data.columns == "CF (Outcome Numeric)"]

#

#
#
list = data['Anon Student Id'].unique()
# print(list)

data.drop(['Anon Student Id'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2)

def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tp = cm[1][1]
    # accuracy = (tp + tn) / (tp + fp + fn + tn + 0.0)
    # precision = tp / (tp + fp + 0.0)
    fprate = fp / (fp + tn + 0.0)
    # True positive rate = recall rate
    tprate = tp / (tp +fn + 0.0)
    print(classifier)
    print("false positive rate is: %0.3f" % fprate)
    print("true positive rate is: %0.3f" % tprate)



def draw_confusion_matrices(confusion_matricies):
    class_names = ['Incorrect', 'Correct']
    for cm in confusion_matricies:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, interpolation='nearest', cmap=plt.get_cmap('Reds'))
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

parameters = {
    'n_estimators': [40, 60, 80]
}
Grid_RF = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
Grid_RF.fit(Xtrain, Ytrain)
best_RF_model = Grid_RF.best_estimator_
parameters = {
    'penalty': ('l1', 'l2'),
    'C': (1, 5, 10)

}
Grid_LR = GridSearchCV(LogisticRegression(), parameters, cv=5)
Grid_LR.fit(Xtrain, Ytrain)
best_LR_model = Grid_LR.best_estimator_
confusion_matrices = [
    ("Random Forest", confusion_matrix(Ytest, best_RF_model.predict(Xtest))),
    ("Logistic Regression", confusion_matrix(Ytest, best_LR_model.predict(Xtest))),
]

draw_confusion_matrices(confusion_matrices)

from sklearn.metrics import roc_curve
from sklearn import metrics

y_pred_rf = best_RF_model.predict_proba(Xtest)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Ytest, y_pred_rf)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()

from sklearn import metrics

# AUC score
metrics.auc(fpr_rf, tpr_rf)

y_pred_lr = best_LR_model.predict_proba(Xtest)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(Ytest, y_pred_lr)

# ROC Curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - LR Model')
plt.legend(loc='best')
plt.show()
metrics.auc(fpr_lr, tpr_lr)
