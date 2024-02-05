import numpy as np
import pandas as pd
import random as rn
from common.config import max_unit_time, random_seed
from sklearn.utils import shuffle
from common.config import label_next, label_next_std, label_diff, label_diff_std, label_current, baseline_std_cols, model_numeric_cols


def get_trending_label(all_data):

    # 构造标签
    all_data["indoor_temp_diff"] = all_data["indoor_temp_next"] - all_data["indoor_temp"]
    # 构造特征
    #all_data["heat_indoor_temp"] = all_data["second_heat_temp"] - all_data["indoor_temp"]
    #all_data["indoor_outdoor_temp"] = all_data["indoor_temp"] - all_data["outdoor_temp"]

    return all_data

def generate_feature(all_data):

    # 构造特征
    all_data["heat_indoor_temp"] = all_data["second_heat_temp"] - all_data["indoor_temp"]
    all_data["indoor_outdoor_temp"] = all_data["indoor_temp"] - all_data["outdoor_temp"]

    return all_data

def generate_next_label(all_data, long_predict_len):
    #  标签
    #  T+1时刻室温原值、室温差值
    # 向上平移一位，T时刻获取T+1时刻的室温
    #all_data["indoor_temp_next"] = all_data["indoor_temp"].shift(-1)
    shift_len = -1 * long_predict_len
    all_data["indoor_temp_next"] = all_data["indoor_temp"].shift(shift_len)
    all_data = all_data.dropna(axis=0, how='any')  # 删除有NAN的行
    all_data = all_data.reset_index(drop=True)  # 重设索引
    return all_data

'''
def get_current_short_long_data(data):

    # 获取T、T-1、T-n的数据
    current_data = data.copy()
    short_data = data.copy()
    long_data = data.copy()
    #short_data['indoor_temp_unit_time'] = 0
    #long_data['indoor_temp_unit_time'] = 0
    #print('long_data', long_data)

    time_length = len(data)
    # test to do
    #for i in range(0, 100): # 0时刻没有T-1数据，从1开始
    for i in range(0, time_length-1): # 0时刻没有T-1数据，从1开始
        if i >= 1:
            current_data.loc[i, :] = data.loc[i, :]
            short_data.loc[i, :] = data.loc[i-1, :]
            # 单位时间为1小时
            short_data.loc[i, 'indoor_temp_unit_time'] = data.loc[i+1, 'indoor_temp']
            # long_data
            sample_max_range = min(i, time_length-i, max_unit_time) # 随机的单位时间应在72小时(max_unit_time)以内，并且T-n和T+n都在数据范围内
            #print('sample_max_range', sample_max_range)
            sampled_n = rn.sample(range(0, sample_max_range), 1)[0]
            if sampled_n == 0: # 单位时间为0时没有做差的意义
                sampled_n = 1
            #print('sampled_n', sampled_n)
            long_data.loc[i, :] = data.loc[i-sampled_n, :]
            # long_data label,单位时间为n小时
            long_data.loc[i, 'indoor_temp_unit_time'] = data.loc[i+sampled_n, 'indoor_temp']
            #print(data.loc[i+sampled_n, 'indoor_temp'])
    
    # 去掉第一行、最后一行的无用数据
    current_data = current_data.drop([0, len(current_data)-1])
    short_data = short_data.drop([0, len(short_data)-1])
    long_data = long_data.drop([0, len(long_data)-1])
    current_data = current_data.reset_index(drop=True)  # 重设索引
    short_data = short_data.reset_index(drop=True)
    long_data = long_data.reset_index(drop=True)

    #print('data', data)
    #print('current_data', current_data)
    #print('short_data', short_data)
    #print('long_data', long_data)

    return current_data, short_data, long_data
'''

def get_current_short_long_data(data):

    time_shift_number = 1
    data_current = data.copy()  # T时刻，当前时刻
    #  short, shift
    data_short = data.copy().shift(time_shift_number)  # 向下平移一位，获取前一时刻的特征
    data_short = data_short.reset_index(drop=True)  # 重设索引
    #  long, shuffle
    # 按行打乱顺序
    data_long = shuffle(data.copy())  #  按行打乱顺序
    data_long = data_long.reset_index(drop=True)  # 重设索引
    #  只需要shift时删除，为保持长度一致，都删除第一行
    data_current.drop(labels=range(0, 1), inplace=True)
    data_short.drop(labels=range(0, 1), inplace=True)
    data_long.drop(labels=range(0, 1), inplace=True)
    data_current = data_current.reset_index(drop=True)  # 重设索引
    data_short = data_short.reset_index(drop=True)
    data_long = data_long.reset_index(drop=True)
    ## diff
    data_current, data_short, data_long = get_diff_label(
        data_current, data_short, data_long)
    # 去掉差值标签的异常值
    #drop_outliers_diff_label_long_short(data_current, data_short, data_long)

    return data_current, data_short, data_long


def get_diff_label(data_current, data_short, data_long):

    #  室温差值，标签
    '''
    #  当前值室温差值：T+1-T时刻
    #  短期室温差值：T+1-T时刻
    #  长期室温差值：T+1时刻-(T-N+1)时刻
    '''
    # 当前值室温差值 和 短期室温差值 一致：T+1-T时刻

    data_current['indoor_temp_diff'] = data_current['indoor_temp_next'] - \
        data_short['indoor_temp_next']
    data_short['indoor_temp_diff'] = data_current['indoor_temp_next'] - \
        data_short['indoor_temp_next']
    data_long['indoor_temp_diff'] = data_current['indoor_temp_next'] - \
        data_long['indoor_temp_next']

    return data_current, data_short, data_long

def get_difference_data(current_data, short_data, long_data):

    # 趋势数据用T时刻特征，加上供温-室温、室温-外温特征。
    #current_data = generate_feature(current_data)
    #short_data = generate_feature(short_data)
    #long_data = generate_feature(long_data)
    #current_data = get_trending_label(current_data) # 使用short_volatility的indoor_temp_diff_unit_time
    #print(current_data.columns.values.tolist())
    current_trending_data = current_data.copy() # 趋势数据使用T时刻数据，趋势也就是真值数据
    short_volatility = get_volatility_label_feature(current_data, short_data) # 波动数据也就是差值数据
    long_volatility = get_volatility_label_feature(current_data, long_data)

    return current_trending_data, short_volatility, long_volatility

def get_volatility_label_feature(current_data, short_long_data):

    volatility_data = pd.DataFrame()
    volatility_data['date_time'] = current_data['time']
    # 构造标签
    #volatility_data['indoor_temp_diff_unit_time'] = short_long_data['indoor_temp_unit_time'] - current_data['indoor_temp'] # short data时等价于all_data["indoor_temp_next"] - all_data["indoor_temp"]
    volatility_data['indoor_temp_diff'] = current_data['indoor_temp_next'] - short_long_data['indoor_temp_next']  # short data时等价于all_data["indoor_temp_next"] - all_data["indoor_temp"]
    # 构造特征
    volatility_data['second_heat_temp_diff'] = current_data['second_heat_temp'] - short_long_data['second_heat_temp']
    #volatility_data['heat_indoor_temp_diff'] = current_data['heat_indoor_temp'] - short_long_data['heat_indoor_temp']
    volatility_data['second_heat_pressure_diff'] = current_data['second_heat_pressure'] - short_long_data['second_heat_pressure']
    volatility_data['illumination_diff'] = current_data['illumination'] - short_long_data['illumination']
    volatility_data['outdoor_temp_diff'] = current_data['outdoor_temp'] - short_long_data['outdoor_temp']
    #volatility_data['indoor_outdoor_temp_diff'] = current_data['indoor_outdoor_temp'] - short_long_data['indoor_outdoor_temp']
    volatility_data['outdoor_pressure_diff'] = current_data['outdoor_pressure'] - short_long_data['outdoor_pressure']
    volatility_data['outdoor_humidity_diff'] = current_data['outdoor_humidity'] - short_long_data['outdoor_humidity']
    volatility_data['wind_speed_diff'] = current_data['wind_speed'] - short_long_data['wind_speed']

    # 离散值
    volatility_data['wind_direction'] = short_long_data['wind_direction']
    volatility_data['weather'] = short_long_data['weather']
    volatility_data['day'] = short_long_data['day']
    volatility_data['hour'] = short_long_data['hour']
    volatility_data['havePeople'] = short_long_data['havePeople']

    #print(volatility_data)
    volatility_data = volatility_data.dropna(axis=0, how='any')
    volatility_data = volatility_data.reset_index(drop=True)
    #print(volatility_data)

    return volatility_data



###########################

def get_multi_time_continuous_data(windows_len, data, cols):

    time_len = len(data[cols].values)
    #print('time_len', time_len)

    data_values = data[cols].values
    #  构造三维数据。第一个维度为样本长度。第二个维度并为时间窗口长度。第三个维度为特征长度
    ## 前windows_len个数据有空值，删掉
    multi_time_data = np.zeros((time_len, windows_len, len(cols)), dtype=np.float)
    # 标签数据也要与特征保持相同时间长度
    label_next_data = data[label_next].values
    label_next_std_data = data[label_next_std].values
    label_current_data = data[label_current].values
    #label_diff_data = data[label_diff].values
    #label_diff_std_data = data[label_diff_std].values

    
    for i in range(0, time_len):

        data_temp = data_values.copy() #  深拷贝，不影响原数据
        # 只能看到之前windows_len个时间片之内的数据，之后的数据看不到
        multi_time_data_temp = np.zeros((windows_len, len(cols)), dtype=np.float)
        # 可以看到当前的数据
        if i < windows_len:
            multi_time_data_temp[windows_len-i-1:windows_len, :] = data_temp[0:i+1, :]  # 部分 train 数据，更前面的数据全部为零
            
            # window_len导致几个0值，需要用后续值，也就是第一个值填充。
            if i < windows_len -1:
                for j in range(0, windows_len-i-1):
                    multi_time_data_temp[j, :] = data_temp[0, :]

        
        else:
            multi_time_data_temp[0:windows_len, :] = data_temp[i-windows_len+1:i+1, :] # 切片左闭右开，[i-windows_len:i)

        multi_time_data[i, :, :] = multi_time_data_temp


        #if i == windows_len:
        #    print('data_temp', data_temp)
        #    print('multi_time_data_temp', multi_time_data_temp)
    '''
    print('data_values\n', data_values)
    print('multi_time_data\n', multi_time_data)
    print('label_next_data\n', label_next_data[0:10])
    print('label_current_data\n', label_current_data)
    '''


    multi_time_label = [label_next_data, label_next_std_data, label_current_data]
    #print('multi_time_label', np.size(multi_time_label_next_data, 0))
    #print('multi_time_data', np.size(multi_time_data, 0))


    return multi_time_label, multi_time_data

def get_multi_time_discrete_data(windows_len, data, col):

    time_len = len(data[col].values)

    data_values = data[col].values
    #  构造三维数据。第一个维度为样本长度。第二个维度并为时间窗口长度。第三个维度为特征长度，离散特征为1，可省略
    ## 前windows_len个数据有空值，删掉
    multi_time_data = np.zeros((time_len, windows_len), dtype=np.int)
    
    for i in range(windows_len, time_len):

        data_temp = data_values.copy() #  深拷贝，不影响原数据
        # 只能看到之前windows_len个时间片之内的数据，当前及之后的数据看不到
        multi_time_data_temp = np.zeros((windows_len), dtype=np.int)
        # 可以看到当前的数据
        if i < windows_len:
            multi_time_data_temp[windows_len-i-1:windows_len] = data_temp[0:i+1]  # 部分 train 数据，更前面的数据全部为零
        else:
            multi_time_data_temp[0:windows_len] = data_temp[i-windows_len+1:i+1] # 切片左闭右开，[i-windows_len:i)


        multi_time_data[i, :] = multi_time_data_temp
        #if i == windows_len:
        #    print('data_temp', data_temp)
        #    print('multi_time_data_temp', multi_time_data_temp)
    
    #print('multi_time_data', multi_time_data)

    return multi_time_data



def get_short_index_array(train_time_len):

    short_index_array = np.array(range(0, train_time_len))
    #print(short_index_array)
    short_index_array = short_index_array - 1
    short_index_array[0] = 0
    #print(short_index_array)

    return short_index_array

def get_long_index_array(train_time_len):

    np.random.seed(random_seed)
    long_index_array = np.random.choice(train_time_len, size=train_time_len, replace=False)

    #print('long_index_array', long_index_array)

    return long_index_array

def get_t_1_t_n_data(multi_time_train_t_0_label, multi_time_train_t_0_data, short_index_array):

    label_next_data, label_next_std_data, label_current_data = multi_time_train_t_0_label
    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_train_t_0_data

    #print('label_next_data', label_next_data)
    label_next_data = label_next_data[short_index_array]
    #print('label_next_data', label_next_data)
    label_next_std_data = label_next_std_data[short_index_array]
    label_current_data = label_current_data[short_index_array]

    multi_time_train_t_0_label = [label_next_data, label_next_std_data, label_current_data]

    continuous_model_data = continuous_model_data[short_index_array][:][:]
    continuous_baseline_data = continuous_baseline_data[short_index_array][:][:]
    wind_data = wind_data[short_index_array][:]
    weather_data = weather_data[short_index_array][:]
    day_data = day_data[short_index_array][:]
    hour_data = hour_data[short_index_array][:]
    havePeople_data = havePeople_data[short_index_array][:]


    multi_time_train_t_0_data = [continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data]

    return multi_time_train_t_0_label, multi_time_train_t_0_data


def get_diff(multi_time_train_t_0_label, multi_time_train_t_0_data, multi_time_train_t_1_label, multi_time_train_t_1_data):

    label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0 = multi_time_train_t_0_label
    label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = multi_time_train_t_1_label

    continuous_model_data_t_0, continuous_baseline_data_t_0, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0 = multi_time_train_t_0_data
    continuous_model_data_t_1, continuous_baseline_data_t_1, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = multi_time_train_t_1_data

    # 标签差
    label_diff = label_next_data_t_0 - label_next_data_t_1

    # 连续数据
    #print('continuous_model_data_t_0\n', continuous_model_data_t_0)
    #print('continuous_model_data_t_1\n', continuous_model_data_t_1)
    continuous_data_diff = continuous_model_data_t_0 - continuous_model_data_t_1
    #print('continuous_data_diff\n', continuous_data_diff)

    short_label = [label_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1]
    short_data = [continuous_data_diff, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]

    return short_label, short_data


def replace_some_time_data(short_data_std_test_new, long_data_std_test_new, short_data_std_test, long_data_std_test, change_time_type, windows_len):

    # 拆开
    continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data_std_test
    continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = long_data_std_test


    continuous_data_short_diff_all_std_new, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data_std_test_new
    continuous_data_long_diff_all_std_new, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = long_data_std_test_new

    continuous_data_short_diff_all_std_part_new = continuous_data_short_diff_all_std.copy()
    continuous_data_long_diff_all_std_part_new = continuous_data_long_diff_all_std.copy()

    continuous_data_short_diff_all_std_part_new[:, windows_len-change_time_type:windows_len, :] = continuous_data_short_diff_all_std_new[:, windows_len-change_time_type:windows_len, :]
    continuous_data_long_diff_all_std_part_new[:, windows_len-change_time_type:windows_len, :] = continuous_data_long_diff_all_std_new[:, windows_len-change_time_type:windows_len, :]

    # 封装
    short_data_std_test_part_new = [continuous_data_short_diff_all_std_part_new, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1]
    long_data_std_test_part_new = [continuous_data_long_diff_all_std_part_new, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n]


    return short_data_std_test_part_new, long_data_std_test_part_new



def get_MDL_day_week_data(train_data, span, windows_len):

    # 获取连续特征，不需要标签数据
    multi_time_label, continuous_baseline_data = get_day_week_continuous_data(windows_len, train_data, baseline_std_cols, span)
    multi_time_label, continuous_model_data = get_day_week_continuous_data(windows_len, train_data, model_numeric_cols, span)

    # 不需要离散特征
    '''
    # 再获取离散特征
    wind_data = get_day_week_discrete_data(windows_len, train_data, 'wind_direction', span)
    weather_data = get_day_week_discrete_data(windows_len, train_data, 'weather', span)
    day_data = get_day_week_discrete_data(windows_len, train_data, 'day', span)
    hour_data = get_day_week_discrete_data(windows_len, train_data, 'hour', span)
    havePeople_data = get_day_week_discrete_data(windows_len, train_data, 'havePeople', span)
    '''


    multi_time_data = [continuous_model_data, continuous_baseline_data]
    
    return multi_time_label, multi_time_data


def get_day_week_continuous_data(windows_len, data, cols, span):


    time_len = len(data[cols].values)

    data_values = data[cols].values

    label_next_data = data[label_next].values
    label_next_std_data = data[label_next_std].values
    label_current_data = data[label_current].values

    multi_time_label = [label_next_data, label_next_std_data, label_current_data]



    #  构造三维数据。第一个维度为样本长度。第二个维度并为时间窗口长度。第三个维度为特征长度
    ## 前windows_len个数据有空值，删掉
    multi_time_data = np.zeros((time_len, windows_len, len(cols)), dtype=np.float)

    data_temp = data_values.copy() #  深拷贝，不影响原数据

    for i in range(0, time_len):

        # 只能看到之前windows_len个时间片之内的数据，之后的数据看不到
        multi_time_data_temp = np.zeros((windows_len, len(cols)), dtype=np.float)

        # 两层循环
        for j in range(0, windows_len):
            data_index = i - j * span # 时间索引，减去时间间隔（天数 * 每天的时间间隔）
            if data_index < 0:
                # 当超出历史数据范围时，索引置零
                data_index = 0
            
            # 赋值
            multi_time_data_temp[j, :] = data_temp[data_index, :]

        multi_time_data[i, :, :] = multi_time_data_temp


    return multi_time_label, multi_time_data

def get_day_week_discrete_data(windows_len, data, cols, span):

    time_len = len(data[cols].values)

    data_values = data[cols].values

    #  构造三维数据。第一个维度为样本长度。第二个维度并为时间窗口长度。第三个维度为特征长度
    ## 前windows_len个数据有空值，删掉
    multi_time_data = np.zeros((time_len, windows_len), dtype=np.float)

    data_temp = data_values.copy() #  深拷贝，不影响原数据

    for i in range(0, time_len):

        # 只能看到之前windows_len个时间片之内的数据，之后的数据看不到
        multi_time_data_temp = np.zeros((windows_len), dtype=np.float)

        # 两层循环
        for j in range(0, windows_len):
            data_index = i - j * span # 时间索引，减去时间间隔（天数 * 每天的时间间隔）
            if data_index < 0:
                # 当超出历史数据范围时，索引置零
                data_index = 0
            
            # 赋值
            multi_time_data_temp[j] = data_temp[data_index]

        multi_time_data[i, :] = multi_time_data_temp


    return multi_time_data


def get_long_short_index_array_enhancement(train_time_len, enhancement_times):

    #np.random.seed(random_seed)
    long_index_array = np.zeros(0, dtype=np.int)
    short_index_array = np.zeros(0, dtype=np.int)
    current_index_array = np.zeros(0, dtype=np.int)

    for i in range(0, train_time_len):
        long_index_array_temp = np.random.choice(train_time_len, size=enhancement_times, replace=False)
        #print('long_index_array_temp', long_index_array_temp)
        long_index_array = np.append(long_index_array, long_index_array_temp)
        #print('long_index_array', long_index_array)
        if i == 0:
            short_index_array_temp = np.ones(enhancement_times).astype(int)  * i
        else:
            short_index_array_temp = np.ones(enhancement_times).astype(int)  * (i-1)
        #print('short_index_array_temp', short_index_array_temp)
        short_index_array = np.append(short_index_array, short_index_array_temp)

        current_index_array_temp = np.ones(enhancement_times).astype(int)  * i
        #print('short_index_array_temp', short_index_array_temp)
        current_index_array = np.append(current_index_array, current_index_array_temp)
        #print('short_index_array', short_index_array)
        #print('i', i)

    #print('short_index_array', short_index_array)
    #print('long_index_array', long_index_array)

    
    return current_index_array, short_index_array, long_index_array


if __name__ == '__main__':

    get_long_short_index_array_enhancement(10, 4)





