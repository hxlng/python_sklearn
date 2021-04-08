import pandas as pd
import numpy as np

# 录入处理好的数据（路径需要自己修改，路径前加r，左斜杠）（需要用什么取消注释哪行）

data_m_train = pd.read_excel(r'C:\Users\Dell\Desktop\西南财大数据建模\data_m_train_one.xlsx')
#data_b_train = pd.read_csv(r'C:\Users\Dell\Desktop\西南财大数据建模\data_b_train.csv')
# y_train = pd.read_csv(r'C:\Users\jion\Desktop\Mooplab比赛数据\y_train.csv')
data_m_test = pd.read_excel(r'C:\Users\Dell\Desktop\西南财大数据建模\data_m_test_one.xlsx')
#data_b_test = pd.read_csv(r'C:\Users\Dell\Desktop\西南财大数据建模\data_b_test.csv')

# 删除行为信息中全为-99的列 x_number_19和x_number_38（输入了行为信息m最好把这行加上）
# data_m_train = data_m_train.drop(columns=['x_num_19', 'x_num_38'])
# data_m_test = data_m_test.drop(columns=['x_num_19', 'x_num_38'])

# 通过输入DataFrame的数据，用均值代替各列-99的方法（替换-99的方法还有很多，可以自己多试试）
def mer_mean_processing(data):
    data_array = np.array(data)  # 将输入的DataFrame数据转化为数组以便计算
    [m, n] = np.shape(data_array)  # 只使用n，提取输入数据总共有多少类型的变量，为计算均值做准备
    res = np.array(data_array)  # 创建一个一摸一样的表，此时有两个表，一个data_array,一个res,后面在data_array上计算，
    # 计算结果存到res，这里如果直接等于的话后面会出错，出错原因大概与pandas的DataFrame有关，总之直接等于的话一个变量改变，另一个会做出相同改变，
    # 达不到我们要的一个计算结果，一个存储结果的目的。
    data_array[data_array == -99] = 0  # 因为要在data_array上面进行计算均值，先把数组中为-99的全部替换成0，以便计算方便
    for i in range(0, n):
        num_1 = np.shape(np.nonzero(data_array[0:, i]))[1]  # 提取第i列非零行的个数
        num_2 = sum(data_array[0:, i])  # 计算第i列的求和值
        data_1 = res[0:, i]  # 暂存第i列
        if(num_1 == 0):
            res[0:, i] = data_1  # 若第i列的非零行个数为0，说明原数据无-99，直接等于暂存的第i列
        else:
            data_1[data_1 == -99] = num_2 * 1.0 / num_1  # 若第i列的非零行个数不为0，说明可能存在-99，先计算均值，再在原数据中替换-99
            res[0:, i] = data_1  # 存储

    # 循环结束，将结果转化为DataFrame，以便使用pandas输出表格
    df_res = pd.DataFrame(res)
    df_res.columns = data.columns
    return df_res

# 均值处理，使用哪个就取消注释哪个，可以查查如何同时输出多个表格到不同文件
data_b_train_treat = mer_mean_processing(data_m_train)
data_b_test_treat = mer_mean_processing(data_m_test)
# data_m_train_treat = mer_mean_processing(data_m_train)
# data_m_test_treat = mer_mean_processing(data_m_test)

# 输出数据（路径需要自己修改，不加r，右斜杠）
writer = pd.ExcelWriter('C:/Users/Dell/Desktop/西南财大数据建模/data_m_train_treat.xlsx')
data_b_train_treat.to_excel(writer, sheet_name='Sheet1')
writer.save()

writer = pd.ExcelWriter('C:/Users/Dell/Desktop/西南财大数据建模/data_m_test_treat.xlsx')
data_b_test_treat.to_excel(writer, sheet_name='Sheet1')
writer.save()
