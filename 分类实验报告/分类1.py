import numpy as np
import pandas as pd
from sklearn import tree
import sklearn.linear_model as lm
import  sklearn.neural_network as nl
import sklearn.svm as s
import sklearn.naive_bayes as nb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


file=["balance-scale","BreastTissue","caesarian","data_banknote_authentication","dataR2","glass","hcvdat0","transfusion","trial","Wholesale customers data"]
a = np.zeros(shape=(10, 5))


for i in range(0,len(file)):
    df_train = pd.read_csv(r"C:\Users\Dell\Desktop\数据集新\{}.csv".format(file[i]))
    target = df_train["label"].values
    Xtrain = df_train.values
    Xtrain = df_train.drop(["label"], axis=1).values
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(target).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=0)

    # 决策树
    M1 = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
    M1.fit(x_train, y_train)

    # Logistics回归
    M2 = lm.LogisticRegression(solver="liblinear", C=1, random_state=0)
    M2.fit(x_train, y_train)

    # BP神经网络
    M3 = nl.MLPClassifier(hidden_layer_sizes=(500,), activation='relu',
                          solver='lbfgs', alpha=0.0001, batch_size='auto',
                          learning_rate='constant', random_state=0)
    M3.fit(x_train, y_train)

    # 支持向量机
    M4 = s.SVC()
    M4.fit(x_train, y_train)

    # 高斯朴素贝叶斯
    M5 = nb.GaussianNB()
    M5.fit(x_train, y_train)

    model=[M1,M2,M3,M4,M5]
    Mm = ['决策树', 'Logistics回归', 'BP神经网络', '支持向量机', '高斯朴素贝叶斯']
    for j in range(0, 5):
        y_predY = model[j].predict(x_test)
        score = float("%.4f"%accuracy_score(y_test, y_predY))
        a[i][j]=score
        print("{} ".format(Mm[j]) + " 精度为：", score)
print(a)
print(a[:,0])




plt.title('五类方法精度对比')  # 折线图标题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('数据集')  # x轴标题
plt.ylabel('精确值')  # y轴标题
x=list(range(len(file)))
for i in range(0,5):
    plt.plot(x, a[:,i], marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小

for a,b in zip(x,a[:,0]):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小

plt.xticks(x,file,rotation=30)  ## 可以设置坐标字

plt.legend(Mm)  # 设置折线名称

plt.show()  # 显示折线图

