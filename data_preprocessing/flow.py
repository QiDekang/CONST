####################################################
# 数据预处理
## 第一步：读取数据，确定保存路径
## 第一步：特征筛选与构造
## 第二步：数据划分（train数据不能使用test时间的+n数据）
## 第三步：标签构造
## 第四步：去除异常值
## 第五步：数据标准化
####################################################
import pandas as pd
from data_preprocessing.data_IO import read_data
from data_preprocessing.drop_outliers import drop_nan_outliers, drop_outliers
from common.config import filter_cols, drop_outliers_clos, baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols, trending_numeric_cols, trending_std_cols
from data_preprocessing.label_feature import generate_next_label, generate_feature, get_current_short_long_data, get_difference_data
from data_preprocessing.split import split_data_no_fold
from data_preprocessing.standard import num_standard_label, get_trending_standard, get_volatility_standard, data_standard_transform, num_standard_feature_transform
from common.config import label_diff, label_diff_std

from common.config import baseline_numeric_cols, baseline_std_cols, model_numeric_cols, model_std_cols, label_next, label_next_std
from data_preprocessing.label_feature import get_multi_time_continuous_data, get_multi_time_discrete_data, get_short_index_array, get_long_index_array, get_t_1_t_n_data, get_diff
from data_preprocessing.standard import get_diff_label_std, get_diff_feature_std, data_standard, get_other_stations_label_diff_std_data
import numpy as np
from STCD_models.modules.temporal_discount import get_temporal_discount_shuffle_data
from data_preprocessing.label_feature import get_MDL_day_week_data
from data_preprocessing.label_feature import get_long_short_index_array_enhancement
from data_preprocessing.drop_outliers import drop_negative



# 数据预处理
def get_data(preparation_floder, preprocessing_floder, file_name, fold_id, standard_type, long_predict_len):

    ## 第一步：读取数据
    all_data = read_data(preparation_floder, file_name)
    #print(len(all_data))

    ## 第二步：去除异常值
    ### 如果供温、室温有一个为空，则删掉数据
    all_data = drop_nan_outliers(all_data)
    #print(len(all_data))
    ### 去掉真值3倍delta外数据
    all_data = drop_outliers(all_data, drop_outliers_clos)
    #print(len(all_data))
    ### 去掉供温减回温为负数的数据。
    all_data = drop_negative(all_data)

    ## 第三步：缺失值填充
    all_data = all_data.fillna(method='ffill', axis = 0) # 每列用前一个值替换缺失值

    ## 第四步：特征筛选、真值标签（下一时刻室温）构造
    all_data = all_data[filter_cols]
    all_data = generate_next_label(all_data, long_predict_len)
    ###  先对室温真值做标准化，用于baselines
    #label_next_ss, all_data = num_standard_label(all_data, "indoor_temp_next", "indoor_temp_next_std", standard_type)
    all_data, label_next_ss, feature_ss = data_standard(all_data, label_next, label_next_std, baseline_numeric_cols, baseline_std_cols, standard_type)

    #print(all_data)
    #all_data.to_csv(preprocessing_floder + 'all_data/' + file_name + '.csv', header=True, index=False)

    ## 第五步：数据划分（train数据不能使用test时间的+n数据）
    ### 6:2:2划分，先取20%作为测试集
    #train_data, test_data = split_data(all_data, 0.2)
    #train_data, test_data = get_train_test_fold(all_data, fold_id)
    train_data, test_data = split_data_no_fold(all_data, 0.2)


    return label_next_ss, feature_ss, train_data, test_data

def get_multi_time_data(all_data, windows_len):

    # 拆分成不同特征，连续特征为4*windows_len或5*windows_len，每个离散特征形状为1*windows_len
    #continuous_baseline_data = all_data[baseline_std_cols]
    #continuous_model_data = all_data[model_std_cols]
    ## 连续特征输入维度 (batch_size, windows_len, 特征长度)
    multi_time_label, continuous_baseline_data = get_multi_time_continuous_data(windows_len, all_data, baseline_std_cols)
    multi_time_label, continuous_model_data = get_multi_time_continuous_data(windows_len, all_data, model_numeric_cols) # 需计算差值后再标准化
    
    
    #print(multi_time_continuous_model_data)
    ## 离散特征输入维度 (batch_size, windows_len)
    wind_data = get_multi_time_discrete_data(windows_len, all_data, 'wind_direction')
    weather_data = get_multi_time_discrete_data(windows_len, all_data, 'weather')
    day_data = get_multi_time_discrete_data(windows_len, all_data, 'day')
    hour_data = get_multi_time_discrete_data(windows_len, all_data, 'hour')
    havePeople_data = get_multi_time_discrete_data(windows_len, all_data, 'havePeople')
    
    multi_time_data = [continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data]
    
    return multi_time_label, multi_time_data



def get_multi_time_data_wo_DF(all_data, windows_len, model_std_cols):

    # 拆分成不同特征，连续特征为4*windows_len或5*windows_len，每个离散特征形状为1*windows_len
    #continuous_baseline_data = all_data[baseline_std_cols]
    #continuous_model_data = all_data[model_std_cols]
    ## 连续特征输入维度 (batch_size, windows_len, 特征长度)
    multi_time_label, continuous_baseline_data = get_multi_time_continuous_data(windows_len, all_data, baseline_std_cols)
    multi_time_label, continuous_model_data = get_multi_time_continuous_data(windows_len, all_data, model_std_cols) # 需计算差值后再标准化
    
    
    #print(multi_time_continuous_model_data)
    ## 离散特征输入维度 (batch_size, windows_len)
    wind_data = get_multi_time_discrete_data(windows_len, all_data, 'wind_direction')
    weather_data = get_multi_time_discrete_data(windows_len, all_data, 'weather')
    day_data = get_multi_time_discrete_data(windows_len, all_data, 'day')
    hour_data = get_multi_time_discrete_data(windows_len, all_data, 'hour')
    havePeople_data = get_multi_time_discrete_data(windows_len, all_data, 'havePeople')
    
    multi_time_data = [continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data]
    
    return multi_time_label, multi_time_data



def get_long_short_data(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len):

    short_index_array = get_short_index_array(train_time_len)
    
    multi_time_train_t_1_label, multi_time_train_t_1_data = get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, short_index_array)
    
    long_index_array = get_long_index_array(train_time_len)

    multi_time_train_t_n_label, multi_time_train_t_n_data = get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, long_index_array)

    return multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, long_index_array

def get_all_diff_data(multi_time_train_t_0_label, multi_time_train_t_0_data, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data):

    short_label, short_data = get_diff(multi_time_train_t_0_label, multi_time_train_t_0_data, multi_time_train_t_1_label, multi_time_train_t_1_data)
    long_label, long_data = get_diff(multi_time_train_t_0_label, multi_time_train_t_0_data, multi_time_train_t_n_label, multi_time_train_t_n_data)

    return short_label, short_data, long_label, long_data

def get_diff_std(short_label, short_data, long_label, long_data, standard_type, windows_len):

    # 长期和短期数据必须使用相同的标准化方法。以长期为准。
    label_short_diff_std, label_long_diff_std, label_diff_ss = get_diff_label_std(short_label, long_label, standard_type)

    short_data_std, long_data_std, feature_diff_ss = get_diff_feature_std(short_data, long_data, standard_type, windows_len)


    return label_short_diff_std, label_long_diff_std, label_diff_ss, short_data_std, long_data_std, feature_diff_ss


# test只标准化特征即可，预测出标签
def get_diff_std_test(short_data, long_data, windows_len, feature_diff_ss):

    #long
    continuous_data_long_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = long_data
    
    continuous_data_long_diff_all_std = continuous_data_long_diff.copy()
    
    for i in range(0, windows_len):
        continuous_data_long_diff_all_std[:, i, :] = feature_diff_ss.transform(continuous_data_long_diff[:, i, :])

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



    return short_data_std, long_data_std


def get_all_other_stations_data(preparation_floder, preprocessing_floder, file_list, file_name, fold_id, label_next_ss, label_diff_ss, feature_ss, feature_diff_ss, train_time_len_target, test_time_len_target, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, trend_effect_rate):

    other_stations_all_data = pd.DataFrame()

    for file_id in range(0, len(file_list)):
        current_file_name = file_list[file_id]
        if current_file_name != file_name:
            # 读取和划分数据
            ## 简单预处理的数据此前已经保存，可直接读取数据。若未处理则需先处理
            ## 为方便对比MDL方法，需要根据目标站点数据标准化其他站点的真值数据
            #current_file_data = pd.read_csv(preprocessing_floder + 'all_data/' + current_file_name + '.csv', index_col=False)
            ## 第一步：读取数据
            all_data = read_data(preparation_floder, file_name)
            ## 第二步：去除异常值
            all_data = drop_nan_outliers(all_data)
            ### 去掉真值3倍delta外数据
            all_data = drop_outliers(all_data, drop_outliers_clos)
            ## 第三步：缺失值填充
            all_data = all_data.fillna(method='ffill', axis = 0) # 每列用前一个值替换缺失值
            ## 第四步：特征筛选、真值标签（下一时刻室温）构造
            all_data = all_data[filter_cols]
            all_data = generate_next_label(all_data, long_predict_len)
            
            # 标准化，用目标站点的标准化信息
            all_data_std = data_standard_transform(all_data, label_next_ss, feature_ss, label_next, label_next_std, baseline_numeric_cols, baseline_std_cols)

            # 合并多个站点的数据
            other_stations_all_data = other_stations_all_data.append(all_data_std)



    ###############
    # 之后的操作都一样，放在循环外
    ###############
    # 拆分数据
    #train_data, test_data = get_train_test_fold(other_stations_all_data, fold_id)
    train_data, test_data = split_data_no_fold(other_stations_all_data, 0.2)

    # 抽样到和目标站点数据一样多
    train_data = train_data.sample(n=train_time_len_target).reset_index(drop=True)
    test_data = test_data.sample(n=test_time_len_target).reset_index(drop=True)

    #print('train_data', train_data)
    train_time_len = np.size(train_data, 0)
    test_time_len = np.size(test_data, 0)

    # multi_data
    multi_time_train_t_0_label, multi_time_train_t_0_data = get_multi_time_data(train_data, windows_len)
    multi_time_test_t_0_label, multi_time_test_t_0_data = get_multi_time_data(test_data, windows_len)

    # t-1, t-n data
    multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, long_index_array = get_long_short_data(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len)
    #enhancement_times = 100
    multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, current_index_array, short_index_array, long_index_array = get_long_short_data_enhancement(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len, enhancement_times)
    multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data, long_index_array_test = get_long_short_data(multi_time_test_t_0_label, multi_time_test_t_0_data, test_time_len)

    # 差分
    short_label, short_data, long_label, long_data = get_all_diff_data(multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data)
    short_label_test, short_data_test, long_label_test, long_data_test = get_all_diff_data(multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data)

    # 使用目标站点数据进行标准化
    short_data_std, long_data_std = get_diff_std_test(short_data, long_data, windows_len, feature_diff_ss)
    short_data_std_test, long_data_std_test = get_diff_std_test(short_data_test, long_data_test, windows_len, feature_diff_ss)
    label_short_diff_std, label_long_diff_std = get_other_stations_label_diff_std_data(short_label, long_label, label_diff_ss)

    # 构造时间差数据和时间贴现数据
    temporal_discount_rate = get_temporal_discount_shuffle_data(train_data, current_index_array, long_index_array, close_effect_rate, periodic_effect_rate, trend_effect_rate)
    current_index_array_test = np.array(range(0, len(test_data)))
    temporal_discount_rate_test = get_temporal_discount_shuffle_data(test_data, current_index_array_test, long_index_array_test, close_effect_rate, periodic_effect_rate, trend_effect_rate)

    # 将有用的数据返回
    train_data_other = [label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, temporal_discount_rate]
    accuracy_data_other = [short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test]
    #multi_time_0_1_n_data_other = [multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data]

    # without difference
    multi_time_0_1_n_data_other = [multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_n_label, multi_time_train_t_n_data, temporal_discount_rate]
    multi_time_0_1_n_data_other_test = [multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test]
    #MDL
    #MDL_data_other_

    return train_data_other, accuracy_data_other, multi_time_0_1_n_data_other, train_data, test_data, multi_time_0_1_n_data_other_test


def get_MDL_data(train_data, train_time_len, long_index_array, windows_len, feature_diff_ss):

    # t-0数据
    continuous_model_day_data_t_0, continuous_model_week_data_t_0, continuous_baseline_day_data, continuous_baseline_week_data = get_MDL_t_0_data(train_data, windows_len)

    MDL_baseline_data = [continuous_baseline_day_data, continuous_baseline_week_data]

    # t-1, t-n数据
    short_index_array = get_short_index_array(train_time_len)
    continuous_model_day_data_t_1, continuous_model_week_data_t_1 = get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, short_index_array)
    continuous_model_day_data_t_n, continuous_model_week_data_t_n = get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, long_index_array)
    

    # diff
    continuous_model_day_short_diff = continuous_model_day_data_t_0 - continuous_model_day_data_t_1
    continuous_model_day_long_diff = continuous_model_day_data_t_0 - continuous_model_day_data_t_n
    continuous_model_week_short_diff = continuous_model_week_data_t_0 - continuous_model_week_data_t_1
    continuous_model_week_long_diff = continuous_model_week_data_t_0 - continuous_model_week_data_t_n

    # 标准化
    continuous_model_day_short_diff_std, continuous_model_day_long_diff_std = get_diff_std_test_MDL(continuous_model_day_short_diff, continuous_model_day_long_diff, windows_len, feature_diff_ss)
    continuous_model_week_short_diff_std, continuous_model_week_long_diff_std = get_diff_std_test_MDL(continuous_model_week_short_diff, continuous_model_week_long_diff, windows_len, feature_diff_ss)

    MDL_model_data_need = [continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std]

    # 其他站点再重新调用即可


    return MDL_baseline_data, MDL_model_data_need


def get_MDL_t_0_data(train_data, windows_len):

    day_span = 24
    #week_span = 189
    week_span = 168
    #day_windows_len = 3
    #week_windows_len = 3
    day_windows_len = windows_len
    week_windows_len = windows_len
    multi_time_day_label, multi_time_day_data = get_MDL_day_week_data(train_data, day_span, day_windows_len)
    #multi_time_week_label, multi_time_week_data = get_MDL_day_week_data(train_data, day_span, day_windows_len)
    multi_time_week_label, multi_time_week_data = get_MDL_day_week_data(train_data, week_span, week_windows_len)

    continuous_model_day_data, continuous_baseline_day_data = multi_time_day_data
    continuous_model_week_data, continuous_baseline_week_data = multi_time_week_data

    # label数据也不需要

    return continuous_model_day_data, continuous_model_week_data, continuous_baseline_day_data, continuous_baseline_week_data



def get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, short_index_array):

    
    continuous_model_day_data_t_1 = continuous_model_day_data_t_0[short_index_array][:][:]
    continuous_model_week_data_t_1 = continuous_model_week_data_t_0[short_index_array][:][:]

    return continuous_model_day_data_t_1, continuous_model_week_data_t_1


def get_diff_std_test_MDL(short_data, long_data, windows_len, feature_diff_ss):

    #long
    #continuous_data_long_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = long_data
    
    long_data_std = long_data.copy()
    
    for i in range(0, windows_len):
        long_data_std[:, i, :] = feature_diff_ss.transform(long_data[:, i, :])

    ##重新封装数据
    #long_data_std = [continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]


    # short
    #continuous_data_short_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data
    #continuous_data_short_diff_std = feature_diff_ss.transform(continuous_data_short_diff)
    short_data_std = short_data.copy()
    
    for i in range(0, windows_len):
        short_data_std[:, i, :] = feature_diff_ss.transform(short_data[:, i, :])

    ##重新封装数据
    #short_data_std = [continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]


    return short_data_std, long_data_std


def get_long_short_data_enhancement(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len, enhancement_times):

    # 数据增强倍数
    current_index_array, short_index_array, long_index_array = get_long_short_index_array_enhancement(train_time_len, enhancement_times)
    #print('long_index_array\n', long_index_array)

    
    multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement = get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, current_index_array)
    
    multi_time_train_t_1_label, multi_time_train_t_1_data = get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, short_index_array)
    
    multi_time_train_t_n_label, multi_time_train_t_n_data = get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, long_index_array)
    #print('multi_time_train_t_n_data\n', multi_time_train_t_n_data)

    return multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, current_index_array, short_index_array, long_index_array


def get_MDL_data_enhancement(train_data, train_time_len, current_index_array, short_index_array, long_index_array, windows_len, feature_diff_ss):

    # t-0数据
    continuous_model_day_data_t_0, continuous_model_week_data_t_0, continuous_baseline_day_data, continuous_baseline_week_data = get_MDL_t_0_data(train_data, windows_len)
    
    # baselines所用数据不需要增强
    MDL_baseline_data = [continuous_baseline_day_data, continuous_baseline_week_data]


    # STCD模型所需数据需要增强
    # t-1, t-n数据
    #short_index_array = get_short_index_array(train_time_len)
    continuous_model_day_data_t_0_enhancement, continuous_model_week_data_t_0_enhancement = get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, current_index_array)
    continuous_model_day_data_t_1, continuous_model_week_data_t_1 = get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, short_index_array)
    continuous_model_day_data_t_n, continuous_model_week_data_t_n = get_MDL_t_1_t_n_data(continuous_model_day_data_t_0, continuous_model_week_data_t_0, long_index_array)
    

    # diff
    continuous_model_day_short_diff = continuous_model_day_data_t_0_enhancement - continuous_model_day_data_t_1
    continuous_model_day_long_diff = continuous_model_day_data_t_0_enhancement - continuous_model_day_data_t_n
    continuous_model_week_short_diff = continuous_model_week_data_t_0_enhancement - continuous_model_week_data_t_1
    continuous_model_week_long_diff = continuous_model_week_data_t_0_enhancement - continuous_model_week_data_t_n

    # 标准化
    continuous_model_day_short_diff_std, continuous_model_day_long_diff_std = get_diff_std_test_MDL(continuous_model_day_short_diff, continuous_model_day_long_diff, windows_len, feature_diff_ss)
    continuous_model_week_short_diff_std, continuous_model_week_long_diff_std = get_diff_std_test_MDL(continuous_model_week_short_diff, continuous_model_week_long_diff, windows_len, feature_diff_ss)

    MDL_model_data_need = [continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std]

    # 其他站点再重新调用即可


    return MDL_baseline_data, MDL_model_data_need

