import pandas as pd
import numpy as np

# pandas 已经废弃Panel，需改用xarray
# 赶时间，先用笨办法。将更高维度拆分为多个二维表
# 计算方向时，同时改变所有的值，避免使用循环
def get_MMF_data(seq_len, train_current, train_short, train_long, test_current, test_short, test_long):
    
    # current_0, short_0, long_0; current_1...
    # t
    train_current_0 = get_history_data(train_current, 0)
    train_short_0 = get_history_data(train_short, 0)
    train_long_0 = get_history_data(train_long, 0)
    test_current_0 = get_history_data(test_current, 0)
    test_short_0 = get_history_data(test_short, 0)
    test_long_0 = get_history_data(test_long, 0)
    # t-1
    train_current_1 = get_history_data(train_current, 1)
    train_short_1 = get_history_data(train_short, 1)
    train_long_1 = get_history_data(train_long, 1)
    test_current_1 = get_history_data(test_current, 1)
    test_short_1 = get_history_data(test_short, 1)
    test_long_1 = get_history_data(test_long, 1)
    # t-2
    train_current_2 = get_history_data(train_current, 1)
    train_short_2 = get_history_data(train_short, 1)
    train_long_2 = get_history_data(train_long, 1)
    test_current_2 = get_history_data(test_current, 1)
    test_short_2 = get_history_data(test_short, 1)
    test_long_2 = get_history_data(test_long, 1)

    MMF_data = [train_current_0, train_short_0, train_long_0, train_current_1, train_short_1, train_long_1, train_current_2, train_short_2, train_long_2, test_current_0, test_short_0, test_long_0, test_current_1, test_short_1, test_long_1, test_current_2, test_short_2, test_long_2]

    return MMF_data

def get_history_data(train_data, time_delay):

    time_length_train = len(train_data)
    #print('time_length_train', time_length_train)
    #print('train_data', train_data)

    history_data = train_data.copy().shift(time_delay)  # 向下平移一位，获取前time_delay时刻的特征
    # 填充nan值
    #print('history_data', history_data)
    history_data = history_data.fillna(method='bfill', axis = 0) # 缺失值出现在前方，每列用后一个值替换缺失值，
    #print('history_data', history_data)

    return history_data