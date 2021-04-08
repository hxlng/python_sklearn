import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Dell\Desktop\数据集\hcvdat0.csv')
# df=df.dropna(how='any')
labels=list(df.columns[4:].values)  #标签
print(labels)
data=df.loc[:,labels]
#means=data['ALB'].mean()
# me=data['ALB'].median() #中位数
# mod=data['ALB'].mode()  #众数

means=[]
for i in labels:
    mean=data[i].mean()
    means.append(round(mean,2))
for k,i in enumerate(labels):
    df[i].fillna(means[k],inplace=True)#NAN 替换成均值
    #df[i].replace(np.nan,means[k],inplace=True)  #NAN 替换成均值

df.to_csv(r"C:\Users\Dell\Desktop\5.csv")
print(df.tail(20))
