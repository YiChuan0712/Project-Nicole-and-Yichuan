# Project-Nicole-and-Yichuan
# 待解决的问题
Preprocessing.py

    ① Child Id 存在大量缺失，直接删除不妥，填补难度大，且Anon Student Id不能很好的对这项进行替代，

    ② Level (Tutor)和Problem Name两项暂时没有得到很好的处理.


NaiveBayes.py（已废弃，见GaussianNB.py，性能更好一些）

    ① 预测准确率较低（83），鉴于代码非常粗糙（手撸的NaiveBayes，没用现成的包所以性能很差），可能含有问题
    

DecisionTree.py

    ① 调参后效果尚可，本身score 91.5（调参前93.6），交叉验证88.5（调参前76），稳定性提升很多
