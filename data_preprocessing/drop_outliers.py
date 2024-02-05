import numpy as np
import pandas as pd

def drop_nan_outliers(samples):

    # 如果供温、室温有一个为空，则删掉数据
    for i in range(0, len(samples)):
        if pd.isnull(samples.loc[i, 'indoor_temp']) or pd.isnull(samples.loc[i, 'second_heat_temp']):
            samples.drop(labels=i, axis=0, inplace=True) # 按行删除，原地替换

    samples = samples.reset_index(drop=True)  # 重设索引

    return samples

import pandas as pd


def drop_outliers(all_data, drop_outliers_clos):

    # print(drop_outliers_clos)
    drop_index_list = []
    for col_name in drop_outliers_clos:
        col_values = all_data[col_name].values
        col_mean = col_values.mean()
        col_std = col_values.std()
        upper_bound = col_mean + 3 * col_std
        Lower_bound = col_mean - 3 * col_std
        #all_data = all_data[(all_data[col_name] > Lower_bound) & (all_data[col_name] < upper_bound)]
        index_list = all_data[(all_data[col_name] < Lower_bound) | (
            all_data[col_name] > upper_bound)].index.tolist()
        # print(index_list)
        if len(index_list) > 0:
            drop_index_list.extend(index_list)
    # 去重
    drop_index_list = list(set(drop_index_list))
    drop_index_list.sort()
    # print(drop_index_list)
    all_data = all_data.drop(labels=drop_index_list, axis=0)
    all_data = all_data.reset_index(drop=True)  # 重设索引

    return all_data

def drop_negative(all_data):

    all_data['heat_return_diff'] = all_data['second_heat_temp'] - all_data['second_return_temp']
    all_data = all_data[all_data['heat_return_diff']>0]
    all_data = all_data.reset_index(drop=True)  # 重设索引

    return all_data