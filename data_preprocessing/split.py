import numpy as np
import pandas as pd
import datetime

def split_data_no_fold(all_data, test_ratio):

    split_len = int((1 - test_ratio) * len(all_data))
    #print(split_len)
    train_data = all_data.iloc[0:split_len, :]
    test_data = all_data.iloc[split_len:len(all_data), :]
    # 重置索引
    train_data = train_data.reset_index(drop=True)  # 重设索引
    test_data = test_data.reset_index(drop=True)  # 重设索引
    #print(train_data)
    #print(test_data)

    return train_data, test_data

'''
def get_train_test_fold(all_data, fold_id):

    # 分割
    # 以左枝的时间为基准划分训练测试数据
    train_data_id, test_data_id = split_data_fold(all_data, fold_id)
    data_train, data_test = split_data(all_data, train_data_id, test_data_id)

    return data_train, data_test
    
def split_data_fold(all_data, fold_id):

    timeLength = all_data.shape[0]
    print(timeLength)
    #train_data = pd.DataFrame(columns=all_data.columns.values.tolist())
    #test_data = pd.DataFrame(columns=all_data.columns.values.tolist())
    #  每月最后七天作为测试集
    #  划分节点
    split_start = datetime.datetime.strptime(
        '2018/11/15 0:00', '%Y/%m/%d %H:%M')
    split_end = datetime.datetime.strptime(
        '2019/03/15 23:59', '%Y/%m/%d %H:%M')
    split_list = []
    for i in range(0, 20):
        delta_days = i * 7
        split_new = split_start + datetime.timedelta(days=delta_days)
        if split_new <= split_end:
            split_list.append(split_new)
    # print(split_list)
    # [0, 3), [3, 4)| [4, 7), [7, 8)| [8, 11), [11, 12)| [12, 15), [15, 16)| [16, 17end)
    # [0, 2), [2, 3)| [3, 6), [6, 7)| [7, 10), [10, 11)| [11, 14), [14, 15)| [15, 17end)
    if fold_id == 0:
        split_one = split_list[3]
        split_two = split_list[4]
        split_three = split_list[7]
        split_four = split_list[8]
        split_five = split_list[11]
        split_six = split_list[12]
        split_seven = split_list[15]
        split_eight = split_list[16]
    elif fold_id == 1:
        split_one = split_list[2]
        split_two = split_list[3]
        split_three = split_list[6]
        split_four = split_list[7]
        split_five = split_list[10]
        split_six = split_list[11]
        split_seven = split_list[14]
        split_eight = split_list[15]
    elif fold_id == 2:
        split_one = split_list[1]
        split_two = split_list[2]
        split_three = split_list[5]
        split_four = split_list[6]
        split_five = split_list[9]
        split_six = split_list[10]
        split_seven = split_list[13]
        split_eight = split_list[14]
    elif fold_id == 3:
        split_one = split_list[0]
        split_two = split_list[1]
        split_three = split_list[4]
        split_four = split_list[5]
        split_five = split_list[8]
        split_six = split_list[9]
        split_seven = split_list[12]
        split_eight = split_list[13]
    #  分割数据
    train_data_id = []
    test_data_id = []
    for i in range(0, timeLength):
        date_time_current = datetime.datetime.strptime(
            all_data["time"].iloc[i], '%Y/%m/%d %H:%M')
        if (split_one <= date_time_current <= split_two) or (split_three <= date_time_current <= split_four) or (split_five <= date_time_current <= split_six) or (split_seven <= date_time_current <= split_eight):
            #test_data = test_data.append(all_data.iloc[i, :], ignore_index=True)
            test_data_id.append(i)
        else:
            # train_data = train_data.append(all_data.iloc[i, :]drop)  #  loc函数：通过行索引 "Index" 中的具体值来取行数据；iloc函数：通过行号来取行数据
            train_data_id.append(i)
    #train_data.to_csv('D:/HeatingOptimization/prediction/result/fold/train_data.csv', header=True)
    #test_data.to_csv('D:/HeatingOptimization/prediction/result/fold/test_data.csv', header=True)

    return train_data_id, test_data_id


def split_data(all_data, train_data_id, test_data_id):

    train_data = all_data.iloc[train_data_id, :]
    test_data = all_data.iloc[test_data_id, :]
    # 重置索引
    train_data = train_data.reset_index(drop=True)  # 重设索引
    test_data = test_data.reset_index(drop=True)  # 重设索引

    return train_data, test_data
'''


