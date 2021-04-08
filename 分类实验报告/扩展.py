import numpy as np
import pandas as pd
import sklearn.neighbors as ne
import xgboost.sklearn as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_train=pd.read_csv(r"C:\Users\Dell\Desktop\分类数据集\transfusion.csv")
target=df_train["whether"].values
Xtrain=df_train.values
Xtrain = df_train.drop(["whether"], axis=1).values
Xtrain=np.array(Xtrain)
Ytrain = np.array(target).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.2,random_state=0)


#k近邻分类
M1=ne.KNeighborsClassifier(n_neighbors=4)
M1.fit(x_train,y_train)

#XGBboost
M2=xgb.XGBClassifier()
M2.fit(x_train, y_train)

y_predY = M1.predict(x_test)
score1 = accuracy_score(y_test, y_predY)
print(" 精度为：",score1)

y_predY = M2.predict(x_test)
score2 = accuracy_score(y_test, y_predY)
print(" 精度为：",score2)