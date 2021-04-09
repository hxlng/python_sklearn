import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#解决数据输出时列名不对齐的问题
pd.set_option('display.unicode.east_asian_width', True)
df=pd.read_excel('data.xlsx',engine='openpyxl')
df=df.apply(pd.to_numeric, errors='ignore') #将表类数据转换成数字类型，忽略不能转换的
#按“射正”列降序排序
# df=df.sort_values(by='射正',ascending=False)
# 顺序排名
# df['顺序排名'] = df['射正'].rank(method="first", ascending=False)
df['进球']=df['进球（点球）'].replace(r'\(\d+\)','',regex=True)
# df1=df['进球（点球）'].replace(regex=r'\(\d+\)',value='') #效果同上句
labels=list(df.columns[4:].values) #取标签
#df[['出场时间','射门','射正','进球']]=df[['出场时间','射门','射正','进球']].astype(int)  #转换类型
df[labels]=df[labels].astype(int) #效果同上句
df.to_excel('data1进球.xlsx',index=False) #写入文件

df1=df.groupby(['球队'])['进球'].sum().reset_index()
#绘制球队进球数*********************************************************
plt.subplot(2,2,1)
x=df1['球队']
y=df1['进球']
# plt.figure(figsize=(8,4))
plt.bar(x,y)
plt.rcParams['font.sans-serif']='SimHei'
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('球队')
plt.ylabel('进球')
plt.title('球队进球图')
plt.tick_params(labelsize = 10)#轴数据字体大小
plt.xticks(rotation=-90)#旋转x轴上文字角度

#绘制*************************************************************************************
df2=df.groupby(['球队'])[['出场次数','出场时间','射门','射正','进球']].sum().reset_index()
df2.to_excel('data2sum.xlsx',index=False) #写入文件
plt.subplot(2,2,3)
# plt.plot(df2['球队'],df2['出场时间'],color='green',label='出场时间',marker='o')
plt.plot(df2['球队'],df2['射门'],color='red',label='射门',marker='s')
plt.plot(df2['球队'],df2['射正'],color='springgreen',label='射正',marker='x')
plt.plot(df2['球队'],df2['进球'],color='skyblue',label='进球',marker='d')
plt.tick_params(labelsize = 10)#轴数据字体大小
plt.xticks(rotation=-90)#旋转x轴上文字角度
plt.legend()
#**************************************************************************
df3=pd.DataFrame()
df3['球队']=df2['球队']
df3['均场时间']=df2['出场时间']/df2['出场次数']
df3['均场射门']=df2['射门']/df2['出场次数']
df3['均场射正']=df2['射正']/df2['出场次数']
df3['均场进球']=df2['进球']/df2['出场次数']
df3['均场进球率']=df3['均场进球']/df3['均场射门']
df3=df3.sort_values(by='均场进球率',ascending=False)
df3['排名顺序']=df3['均场进球率'].rank(method="first",ascending=False)
df3.to_excel('datajunchang.xlsx',index=False)
#球队排名****************
plt.subplot(2,2,4)
plt.plot(df3['球队'],df3['均场射门'],color='fuchsia',label='均场射门',marker='s')
plt.plot(df3['球队'],df3['均场射正'],color='pink',label='均场射正',marker='x')
plt.plot(df3['球队'],df3['均场进球'],color='m',label='均场进球',marker='d')
plt.xlabel('球队综合实力从前往后递减')
# plt.plot(df3['球队'],df3['排名顺序'],color='green',label='排名',marker='o')
plt.tick_params(labelsize=10)  # 轴数据字体大小
plt.xticks(rotation=-90)  # 旋转x轴上文字角度
plt.legend()
#**************************************************************
#球员排名
plt.subplot(2,2,2)
df4=pd.DataFrame()
df4['球员']=df['球员']
df4['均场进球率']=df['进球']/df['出场次数']
df4=df4.groupby(['球员'])['均场进球率'].sum().reset_index()
plt.plot(df4['球员'],df4['均场进球率'],color='lightgreen',label='球员均场进球',marker='s')
plt.tick_params(labelsize=10)  # 轴数据字体大小
plt.xticks(rotation=-90)  # 旋转x轴上文字角度
plt.legend()
plt.show()


