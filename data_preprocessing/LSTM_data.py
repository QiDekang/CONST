import pandas as pd
import numpy as np


######  设置窗口大小，只传输固定长度的数据。

####  使用numpy而不是pandas可以只传递有用的部分
# （1）减小内存。
# （2）同时，不需要在fit、Predict阶段分别构造history_input、history_output。一次性构造完。网络不接受pandas三维Panel数据。
# （3）pd.Panel在新版本中已弃用
#### 三维矩阵，历史信息长度需一致，均为windows_len。用零补齐：某时刻只能看到之前时刻的数据，该时刻和之后时刻的数据都不能看到
def get_train_history_data(windows_len, train_data, history_cols):

    time_length_train = len(train_data["second_heat_temp"].values)

    # train阶段不能看到test数据
    all_data_values = train_data[history_cols].values
    #  构造三维数据。第一个维度和train_data对应，长度train_len。第二个维度是t时刻的历史信息，为0到t-1时刻的数据，t时刻及之后数据置零，并且总长度为windows_len，保持长度一致。第三个维度为特征长度
    train_history_data = np.zeros((time_length_train, windows_len, len(history_cols)), dtype=np.float)
    
    for i in range(0, time_length_train):

        all_data_temp = all_data_values.copy() #  深拷贝，不影响原数据
        # 只能看到之前windows_len个时间片之内的数据，当前及之后的数据看不到
        history_data_temp = np.zeros((windows_len, len(history_cols)), dtype=np.float)
        # lstm data, 可以看到当前的数据
        if i < windows_len:
            history_data_temp[windows_len-i-1:windows_len, :] = all_data_temp[0:i+1, :]  # 部分 train 数据，更前面的数据全部为零
        if i >= windows_len:
            history_data_temp[0:windows_len, :] = all_data_temp[i-windows_len+1:i+1, :] # 切片左闭右开，[i-windows_len:i)

        train_history_data[i, :, :] = history_data_temp
    
    #print('train_history_data', train_history_data)

    return train_history_data

def get_test_history_data(windows_len, train_data, test_data, history_cols):

    time_length_test = len(test_data["second_heat_temp"].values)
    time_length_train = len(train_data["second_heat_temp"].values)

    # test阶段能看到train数据
    test_data_values = test_data[history_cols].values
    train_data_values = train_data[history_cols].values
    #  构造三维数据。第一个维度和test_data对应，长度test_len。第二个维度是t时刻的历史信息，为0到t-1时刻的数据，t时刻及之后数据置零，并且总长度为windows_len，保持长度一致。第三个维度为特征长度
    test_history_data = np.zeros((time_length_test, windows_len, len(history_cols)), dtype=np.float)
    
    for i in range(0, time_length_test):

        # 只能看到之前windows_len个时间片之内的数据，当前及之后的数据看不到
        history_data_temp = np.zeros((windows_len, len(history_cols)), dtype=np.float)
        # 小于时间窗口，从train数据中选取历史数据
        if i < windows_len:
            test_data_temp = test_data_values.copy() #  深拷贝，不影响原数据
            train_data_temp = train_data_values.copy()
            history_data_temp[0:windows_len-i-1, :] = train_data_temp[time_length_train-windows_len+i+1:time_length_train, :] # 部分 train 数据
            history_data_temp[windows_len-i-1:windows_len, :] = test_data_temp[0:i+1, :]  # 部分 test 数据

        # 大于等于时间窗口，从test数据中选取历史数据
        if i >= windows_len:
            test_data_temp = test_data_values.copy() #  深拷贝，不影响原数据
            history_data_temp[0:windows_len, :] = test_data_temp[i-windows_len+1:i+1, :]

            
        test_history_data[i, :, :] = history_data_temp
    
    #print('test_history_data', test_history_data)

    return test_history_data


####  lstm data
def get_lstm_data(windows_len, train_current, train_short, train_long, test_current, test_short, test_long, feature_std_cols, label_std_col):

    ## feature
    #  train
    train_lstm_current = get_train_history_data(windows_len, train_current, feature_std_cols)

    #  test
    test_lstm_current = get_test_history_data(windows_len, train_current, test_current, feature_std_cols)

    ## label
    train_lstm_label = train_current[label_std_col].values
    test_lstm_label = test_current[label_std_col].values

    return train_lstm_label, train_lstm_current, test_lstm_label, test_lstm_current
