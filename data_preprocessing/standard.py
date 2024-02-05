import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from common.config import label_diff, label_diff_std


def get_trending_standard(train_trending_data, test_trending_data, trending_numeric_cols, trending_std_cols, standard_type):

    trending_feature_ss, train_trending_data = num_standard_feature(train_trending_data, trending_numeric_cols, trending_std_cols, standard_type)
    test_trending_data = num_standard_feature_transform(test_trending_data, trending_feature_ss, trending_numeric_cols, trending_std_cols)

    return trending_feature_ss, train_trending_data, test_trending_data

def get_volatility_standard(train_short_volatility, train_long_volatility, test_short_volatility, test_long_volatility, volatility_numeric_cols, volatility_std_cols, standard_type):
    
    label_col = label_diff
    label_col_std = label_diff_std
    # label、feature以long_data为准。
    train_long_volatility, label_diff_ss, volatility_feature_ss=data_standard(
        train_long_volatility, label_col, label_col_std, volatility_numeric_cols, volatility_std_cols, standard_type)
    train_short_volatility = data_standard_transform(
        train_short_volatility, label_diff_ss, volatility_feature_ss, label_col, label_col_std, volatility_numeric_cols, volatility_std_cols)
    test_long_volatility = data_standard_transform(
        test_long_volatility, label_diff_ss, volatility_feature_ss, label_col, label_col_std, volatility_numeric_cols, volatility_std_cols)
    test_short_volatility = data_standard_transform(
        test_short_volatility, label_diff_ss, volatility_feature_ss, label_col, label_col_std, volatility_numeric_cols, volatility_std_cols)

    return label_diff_ss, volatility_feature_ss, train_short_volatility, train_long_volatility, test_short_volatility, test_long_volatility


# standard

def data_standard(data, label_col, label_col_std, numeric_cols, std_cols, standard_type):

    label_next_ss, data_std = num_standard_label(
        data, label_col, label_col_std, standard_type)
    feature_ss, data_std = num_standard_feature(
        data_std, numeric_cols, std_cols, standard_type)

    return data_std, label_next_ss, feature_ss


def num_standard_label(all_data, label_col, label_col_std, standard_type):

    label_std, label_ss = num_standard_base(
        all_data[label_col].values.reshape(-1, 1), standard_type)
    label_std = pd.Series(label_std.flatten())
    all_data[label_col_std] = label_std

    return label_ss, all_data


def num_standard_feature(all_data, numeric_cols, std_cols, standard_type):

    feature_numeric = all_data[numeric_cols]
    #print('feature_numeric', feature_numeric)
    feature_std, feature_ss = num_standard_base(feature_numeric, standard_type)
    #print('feature_std', feature_std)
    feature_std = pd.DataFrame(feature_std, columns=std_cols)
    all_data_std = pd.concat([all_data, feature_std], axis=1)  # 原数据和和标准化的数据拼接

    return feature_ss, all_data_std


def num_standard_base(all_data_ori, standard_type):

    if standard_type == 'minmax':
        ss = MinMaxScaler()
    else:
        ss = StandardScaler()
    data_std = ss.fit_transform(all_data_ori)

    return data_std, ss

# 使用已有StandardScaler对其他数据standard


def data_standard_transform(data, label_next_ss, feature_ss, label_col, label_col_std, numeric_cols, std_cols):

    data_std = num_standard_label_transform(
        data, label_next_ss, label_col, label_col_std)
    data_std = num_standard_feature_transform(
        data_std, feature_ss, numeric_cols, std_cols)

    return data_std


def num_standard_label_transform(all_data, label_ss, label_col, label_col_std):

    label_std = label_ss.transform(all_data[label_col].values.reshape(-1, 1))
    label_std = pd.Series(label_std.flatten())
    all_data[label_col_std] = label_std

    return all_data


def num_standard_feature_transform(all_data, feature_ss, numeric_cols, std_cols):

    feature_numeric = all_data[numeric_cols]
    feature_std = feature_ss.transform(feature_numeric)
    feature_std = pd.DataFrame(feature_std, columns=std_cols)
    all_data_std = pd.concat([all_data, feature_std], axis=1)  # 原数据和和标准化的数据拼接

    return all_data_std


def get_direction_standard(test_trending_data, test_short_volatility, test_long_volatility, trending_feature_ss, volatility_feature_ss, baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols):

    test_trending_data = num_standard_feature_transform(test_trending_data, trending_feature_ss, baseline_trending_numeric_cols, baseline_trending_std_cols)
    test_short_volatility = num_standard_feature_transform(test_short_volatility, volatility_feature_ss, volatility_numeric_cols, volatility_std_cols)
    test_long_volatility = num_standard_feature_transform(test_long_volatility, volatility_feature_ss, volatility_numeric_cols, volatility_std_cols)

    return test_trending_data, test_short_volatility, test_long_volatility


## 用于LSTM
def num_standard_feature_transform_drop(all_data, feature_ss, numeric_cols, std_cols):

    all_data = all_data.drop(labels=std_cols, axis=1)  # 先删除标准化的列
    feature_numeric = all_data[numeric_cols]
    feature_std = feature_ss.transform(feature_numeric)
    feature_std = pd.DataFrame(feature_std, columns=std_cols)
    all_data_std = pd.concat([all_data, feature_std], axis=1)  # 原数据和和标准化的数据拼接

    return all_data_std


def get_diff_label_std(short_label, long_label, standard_type):

    # 以长期数据为准
    label_long_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = long_label

    ## 标签数据标准化
    label_long_diff_std, label_diff_ss = num_standard_base(label_long_diff.reshape(-1, 1), standard_type)

    ## 一列
    label_long_diff_std = label_long_diff_std.reshape(np.size(label_long_diff_std, 0))
    #print('label_long_diff_std\n', label_long_diff_std)

    # 再处理短期标签
    label_short_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label

    label_short_diff_std = label_diff_ss.transform(label_short_diff.reshape(-1, 1))
    label_short_diff_std = label_short_diff_std.reshape(np.size(label_short_diff_std, 0))


    return label_short_diff_std, label_long_diff_std, label_diff_ss


def get_diff_feature_std(short_data, long_data, standard_type, windows_len):

    # long
    ## 未严格改下t_1为t_n
    continuous_data_long_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = long_data
    
    ## 标准化只能处理2维数组
    # 取出第一个时刻的数据
    continuous_data_long_diff_current = continuous_data_long_diff[:, 0, :]
    #continuous_data_long_diff_current = continuous_data_long_diff[:][0][:]
    #print('continuous_data_long_diff\n', continuous_data_long_diff)
    #print('continuous_data_long_diff_current\n', continuous_data_long_diff_current)
    continuous_data_long_diff_current_std, feature_diff_ss = num_standard_base(continuous_data_long_diff_current, standard_type)
    #print('continuous_data_long_diff_current_std \n', continuous_data_long_diff_current_std)


    # 先创建相同大小多维数组
    continuous_data_long_diff_all_std = continuous_data_long_diff.copy()
    continuous_data_long_diff_all_std[:, 0, :] = continuous_data_long_diff_current_std

    for i in range(1, windows_len):
        #globals()['continuous_data_long_diff_std' + str(i)] = continuous_data_long_diff[:][i][:]
        #globals()['continuous_data_long_diff_std' + str(i)] = feature_diff_ss.transform(continuous_data_long_diff[:][i][:])
        continuous_data_long_diff_all_std[:, i, :] = feature_diff_ss.transform(continuous_data_long_diff[:, i, :])


    #print('continuous_data_long_diff_all_std\n', continuous_data_long_diff_all_std)

    ##重新封装数据
    long_data_std = [continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]


    # short
    continuous_data_short_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data
    #continuous_data_short_diff_std = feature_diff_ss.transform(continuous_data_short_diff)
    continuous_data_short_diff_all_std = continuous_data_short_diff.copy()
    
    for i in range(0, windows_len):
        continuous_data_short_diff_all_std[:, i, :] = feature_diff_ss.transform(continuous_data_short_diff[:, i, :])

    ##重新封装数据
    short_data_std = [continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]

    return short_data_std, long_data_std, feature_diff_ss


def get_other_stations_label_diff_std_data(short_label, long_label, label_diff_ss):

    # 再处理短期标签
    label_short_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label

    label_short_diff_std = label_diff_ss.transform(label_short_diff.reshape(-1, 1))
    label_short_diff_std = label_short_diff_std.reshape(np.size(label_short_diff_std, 0))

    # 再处理长期标签
    label_long_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label

    label_long_diff_std = label_diff_ss.transform(label_long_diff.reshape(-1, 1))
    label_long_diff_std = label_long_diff_std.reshape(np.size(label_long_diff_std, 0))


    return label_short_diff_std, label_long_diff_std







