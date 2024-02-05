#  跨文件调用
import numpy as np
import pandas as pd
import datetime
#from common.config import close_effect_rate, periodic_effect_rate
import math

'''
def get_temporal_discount_data(train_t_0_data, sample_array):

    time_diff_close_day_sample, time_diff_periodic_hour_sample = get_time_diff_data(train_t_0_data, sample_array)
    #print('time_diff_close_day_sample\n', time_diff_close_day_sample)
    #print('time_diff_periodic_hour_sample\n', time_diff_periodic_hour_sample)
    temporal_discount_rate = get_temporal_discount_rate(time_diff_close_day_sample, time_diff_periodic_hour_sample, close_effect_rate, periodic_effect_rate)
    #print('temporal_discount_rate\n', temporal_discount_rate)

    return temporal_discount_rate

def get_time_diff_data(train_t_0_data, sample_array):

    time_len = np.size(train_t_0_data, 0)
    # i代表t时刻，j代表t'时刻
    time_diff_close_day = np.zeros((time_len * time_len), dtype=np.int)
    time_diff_periodic_hour = np.zeros((time_len * time_len), dtype=np.int)

    train_t_0_data_copy = train_t_0_data.copy()
    train_t_0_data_copy['date_time'] = pd.to_datetime(train_t_0_data_copy['time'])

    for i in range(0, time_len):
        for j in range(0, time_len):
            pairwise_index = i * time_len + j
            #time_i = datetime.datetime.strptime(train_t_0_data.loc[i, 'time'], '%Y/%m/%d %H:%M')
            #time_j = datetime.datetime.strptime(train_t_0_data.loc[j, 'time'], '%Y/%m/%d %H:%M')
            time_i = train_t_0_data_copy.loc[i, 'date_time']
            time_j = train_t_0_data_copy.loc[j, 'date_time']
            if time_i > time_j:
                timeSpan = time_i - time_j
            else:
                timeSpan = time_j - time_i
            #print('timeSpan', timeSpan)
            time_diff_close_day[pairwise_index] = int(timeSpan.days) # 相差天数
            time_diff_periodic_hour[pairwise_index] = int(timeSpan.seconds / 3600)  #一天内相差小时数。时间向下取整，类似18:01的数据会视为18:00
    

    # 抽样
    time_diff_close_day_sample = time_diff_close_day[sample_array]
    time_diff_periodic_hour_sample = time_diff_periodic_hour[sample_array]
    
    return time_diff_close_day_sample, time_diff_periodic_hour_sample

def get_temporal_discount_rate(time_diff_close_day_sample, time_diff_periodic_hour_sample, close_effect_rate, periodic_effect_rate):

    time_len = np.size(time_diff_close_day_sample, 0)
    temporal_close_discount_rate = np.zeros((time_len), dtype=np.float)
    temporal_periodic_discount_rate = np.zeros((time_len), dtype=np.float)
    for i in range(0, time_len):
        time_diff_close = time_diff_close_day_sample[i]
        temporal_close_discount_rate[i] = round(math.pow(close_effect_rate, time_diff_close), 4)
        time_diff_periodic = time_diff_periodic_hour_sample[i]
        if time_diff_periodic < 12:
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic), 4)
        else:
            time_diff_periodic_new = abs(time_diff_periodic - 24)
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic_new), 4)

    temporal_discount_rate = temporal_close_discount_rate * temporal_periodic_discount_rate

    return temporal_discount_rate


def get_temporal_discount_data_no_sample(train_t_0_data):

    time_diff_close_day_sample, time_diff_periodic_hour_sample = get_time_diff_data_no_sample(train_t_0_data)
    #print('time_diff_close_day_no_sample\n', time_diff_close_day_sample)
    #print('time_diff_periodic_hour_no_sample\n', time_diff_periodic_hour_sample)
    temporal_discount_rate = get_temporal_discount_rate_no_sample(time_diff_close_day_sample, time_diff_periodic_hour_sample, close_effect_rate, periodic_effect_rate)
    #print('temporal_discount_rate no_sample\n', temporal_discount_rate)

    return temporal_discount_rate

def get_time_diff_data_no_sample(train_t_0_data):

    time_len = np.size(train_t_0_data, 0)
    # i代表t时刻，j代表t'时刻
    time_diff_close_day = np.zeros((time_len * time_len), dtype=np.int)
    time_diff_periodic_hour = np.zeros((time_len * time_len), dtype=np.int)

    train_t_0_data_copy = train_t_0_data.copy()
    train_t_0_data_copy['date_time'] = pd.to_datetime(train_t_0_data_copy['time'])

    for i in range(0, time_len):
        for j in range(0, time_len):
            pairwise_index = i * time_len + j
            #time_i = datetime.datetime.strptime(train_t_0_data.loc[i, 'time'], '%Y/%m/%d %H:%M')
            #time_j = datetime.datetime.strptime(train_t_0_data.loc[j, 'time'], '%Y/%m/%d %H:%M')
            time_i = train_t_0_data_copy.loc[i, 'date_time']
            time_j = train_t_0_data_copy.loc[j, 'date_time']
            if time_i > time_j:
                timeSpan = time_i - time_j
            else:
                timeSpan = time_j - time_i
            #print('timeSpan', timeSpan)
            time_diff_close_day[pairwise_index] = int(timeSpan.days) # 相差天数
            time_diff_periodic_hour[pairwise_index] = int(timeSpan.seconds / 3600)  #一天内相差小时数。时间向下取整，类似18:01的数据会视为18:00
    

    # 抽样
    #time_diff_close_day_sample = time_diff_close_day[sample_array]
    #time_diff_periodic_hour_sample = time_diff_periodic_hour[sample_array]
    
    return time_diff_close_day, time_diff_close_day

def get_temporal_discount_rate_no_sample(time_diff_close_day_sample, time_diff_periodic_hour_sample, close_effect_rate, periodic_effect_rate):

    time_len = np.size(time_diff_close_day_sample, 0)
    temporal_close_discount_rate = np.zeros((time_len), dtype=np.float)
    temporal_periodic_discount_rate = np.zeros((time_len), dtype=np.float)
    for i in range(0, time_len):
        time_diff_close = time_diff_close_day_sample[i]
        temporal_close_discount_rate[i] = round(math.pow(close_effect_rate, time_diff_close), 4)
        time_diff_periodic = time_diff_periodic_hour_sample[i]
        if time_diff_periodic < 12:
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic), 4)
        else:
            time_diff_periodic_new = abs(time_diff_periodic - 24)
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic_new), 4)

    temporal_discount_rate = temporal_close_discount_rate * temporal_periodic_discount_rate

    return temporal_discount_rate

'''



def get_temporal_discount_shuffle_data(train_t_0_data, current_index_array, shuffle_array, close_effect_rate, periodic_effect_rate, trend_effect_rate):

    time_diff_close_day_sample, time_diff_periodic_hour_sample, time_diff_trend_week_sample = get_time_diff_shuffle_data(train_t_0_data, current_index_array, shuffle_array)
    #print('time_diff_close_day_sample\n', time_diff_close_day_sample)
    #print('time_diff_periodic_hour_sample\n', time_diff_periodic_hour_sample)

    ################################
    # 不需要给short分支乘以贴现，因为会使得test时，在模型结果后乘以贴现，而真实的Y不会乘以贴现。
    # 为了处理short分支未乘，仅long分支乘贴现，导致的不对称问题。将大于0的时间差统一减1，这样两个分支都乘以了贴现，只是因为短期乘以的贴现中时间差均为0，任何数的0次幂都是1。则短期分支相当于没有乘贴现。
    ################################
    # 短期特征T和T-1均相差一个小时，也就是close_day均为0，periodic_hour均为1
    #time_diff_close_day_sample_short = np.ones(len(time_diff_close_day_sample), dtype=np.int)
    #time_diff_periodic_hour_sample_short = np.zeros(len(time_diff_periodic_hour_sample), dtype=np.int)

    temporal_discount_rate = get_temporal_discount_shuffle_rate(time_diff_close_day_sample, time_diff_periodic_hour_sample, time_diff_trend_week_sample, close_effect_rate, periodic_effect_rate, trend_effect_rate)
    #print('temporal_discount_rate\n', temporal_discount_rate)

    return temporal_discount_rate

def get_time_diff_shuffle_data(train_t_0_data, current_index_array, sample_array):

    #time_len = np.size(train_t_0_data, 0)
    time_len = np.size(sample_array, 0)
    # i代表t时刻，j代表t'时刻
    time_diff_close_day = np.zeros((time_len), dtype=np.int)
    time_diff_periodic_hour = np.zeros((time_len), dtype=np.int)
    time_diff_trend_week = np.zeros((time_len), dtype=np.int)

    train_t_0_data_copy = train_t_0_data.copy()
    train_t_0_data_copy['date_time'] = pd.to_datetime(train_t_0_data_copy['time'])

    for i in range(0, time_len):
        #time_i = datetime.datetime.strptime(train_t_0_data.loc[i, 'time'], '%Y/%m/%d %H:%M')
        #time_j = datetime.datetime.strptime(train_t_0_data.loc[j, 'time'], '%Y/%m/%d %H:%M')
        index_current = current_index_array[i]
        index_long = sample_array[i]

        time_i = train_t_0_data_copy.loc[index_current, 'date_time']
        time_j = train_t_0_data_copy.loc[index_long, 'date_time']
        if time_i > time_j:
            timeSpan = time_i - time_j
        else:
            timeSpan = time_j - time_i
        #print('timeSpan', timeSpan)
        time_diff_close_day[i] = int(timeSpan.days) # 相差天数, 向下取整int()
        time_diff_trend_week[i] = int(timeSpan.days / 7) # 相差周数, 向下取整int()

        ################################
        # 不需要给short分支乘以贴现，因为会使得test时，在模型结果后乘以贴现，而真实的Y不会乘以贴现。
        # 为了处理short分支未乘，仅long分支乘贴现，导致的不对称问题。将大于0的时间差统一减1，这样两个分支都乘以了贴现，只是因为短期乘以的贴现中时间差均为0，任何数的0次幂都是1。则短期分支相当于没有乘贴现。
        ################################
        #time_diff_periodic_hour[i] = int(timeSpan.seconds / 3600)  #一天内相差小时数。时间向下取整，类似18:01的数据会视为18:00
        hour_diff = int(timeSpan.seconds / 3600)  #一天内相差小时数。时间向下取整，类似18:01的数据会视为18:00
        if hour_diff > 0:
            hour_diff = hour_diff - 1
        time_diff_periodic_hour[i] = hour_diff

    #print('time_diff_close_day', time_diff_close_day)
    #print('time_diff_periodic_hour', time_diff_periodic_hour)
    #print('time_diff_trend_week', time_diff_trend_week)
    
    return time_diff_close_day, time_diff_periodic_hour, time_diff_trend_week

'''
def get_time_diff_shuffle_data(train_t_0_data, sample_array):

    #time_len = np.size(train_t_0_data, 0)
    time_len = np.size(sample_array, 0)
    # i代表t时刻，j代表t'时刻
    time_diff_close_day = np.zeros((time_len), dtype=np.int)
    time_diff_periodic_hour = np.zeros((time_len), dtype=np.int)

    train_t_0_data_copy = train_t_0_data.copy()
    train_t_0_data_copy['date_time'] = pd.to_datetime(train_t_0_data_copy['time'])

    for i in range(0, time_len):
        #time_i = datetime.datetime.strptime(train_t_0_data.loc[i, 'time'], '%Y/%m/%d %H:%M')
        #time_j = datetime.datetime.strptime(train_t_0_data.loc[j, 'time'], '%Y/%m/%d %H:%M')
        j = sample_array[i]
        time_i = train_t_0_data_copy.loc[i, 'date_time']
        time_j = train_t_0_data_copy.loc[j, 'date_time']
        if time_i > time_j:
            timeSpan = time_i - time_j
        else:
            timeSpan = time_j - time_i
        #print('timeSpan', timeSpan)
        time_diff_close_day[i] = int(timeSpan.days) # 相差天数
        time_diff_periodic_hour[i] = int(timeSpan.seconds / 3600)  #一天内相差小时数。时间向下取整，类似18:01的数据会视为18:00

    
    return time_diff_close_day, time_diff_periodic_hour
'''

def get_temporal_discount_shuffle_rate(time_diff_close_day_sample, time_diff_periodic_hour_sample, time_diff_trend_week_sample, close_effect_rate, periodic_effect_rate, trend_effect_rate):

    time_len = np.size(time_diff_close_day_sample, 0)
    temporal_close_discount_rate = np.zeros((time_len), dtype=np.float)
    temporal_trend_discount_rate = np.zeros((time_len), dtype=np.float)
    temporal_periodic_discount_rate = np.zeros((time_len), dtype=np.float)
    for i in range(0, time_len):
        time_diff_close = time_diff_close_day_sample[i]
        temporal_close_discount_rate[i] = round(math.pow(close_effect_rate, time_diff_close), 4)

        time_diff_trend = time_diff_trend_week_sample[i]
        temporal_trend_discount_rate[i] = round(math.pow(trend_effect_rate, time_diff_trend), 4)

        time_diff_periodic = time_diff_periodic_hour_sample[i]
        if time_diff_periodic < 12:
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic), 4)
        else:
            time_diff_periodic_new = abs(time_diff_periodic - 24)
            temporal_periodic_discount_rate[i] = round(math.pow(periodic_effect_rate, time_diff_periodic_new), 4)

    #temporal_discount_rate = np.round(temporal_close_discount_rate * temporal_periodic_discount_rate, 4)
    temporal_discount_rate = np.round(temporal_close_discount_rate * temporal_periodic_discount_rate * temporal_trend_discount_rate, 4)
    
    #print('temporal_discount_rate', temporal_discount_rate)

    return temporal_discount_rate







