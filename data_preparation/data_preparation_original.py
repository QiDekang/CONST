import os
import csv
import datetime
import numpy as np
import pandas as pd

def wind_direction_code(ori_code):
    if pd.isnull(ori_code):
        #return_code = -999
        return_code = ori_code
    else:
        ori_code = int(ori_code)
        if ori_code == 0:
            return_code = 0
        if ori_code == 1:
            return_code = 1
        if ori_code == 13:
            return_code = 2
        if ori_code == 3:
            return_code = 3
        if ori_code == 23:
            return_code = 4
        if ori_code == 2:
            return_code = 5
        if ori_code == 24:
            return_code = 6
        if ori_code == 4:
            return_code = 7
        if ori_code == 14:
            return_code = 8
    
    return return_code


def Weather_code(ori_code):
    if pd.isnull(ori_code):
        #return_code = -999
        return_code = ori_code
    else:
        ori_code = int(ori_code)
        if ori_code == 0:
            return_code = 0
        if ori_code == 1:
            return_code = 1
        if ori_code == 2:
            return_code = 2
        if ori_code >= 3 and ori_code <= 9:  # 雨
            return_code = 3
        if ori_code >= 10 and ori_code <= 13:  # 雪
            return_code = 4
        if ori_code >= 14:  # 雾、沙尘
            return_code = 5

    return return_code

def initialization():

    # 初始化
    start_time = '2018/11/15 0:00:00'
    end_time = '2019/03/15 23:59:59'
    date_start = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    date_end = datetime.datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
    time_length = 2904
    cols = ['time', 'indoor_temp', 'indoor_humidity', 'second_heat_temp', 'second_return_temp', 'second_heat_pressure', 'second_return_pressure', 'illumination', 'outdoor_temp', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed', 'wind_direction', 'weather', 'day', 'hour', 'havePeople']
    #samples_zero = np.ones((time_length, len(cols)), dtype=np.float)
    #samples_zero = -999 * samples_zero
    #samples = pd.DataFrame(samples_zero, columns=cols)
    samples = pd.DataFrame(columns=cols)

    return date_start, date_end, time_length, samples

def get_time_data(samples, date_start, date_end, time_length):

    #  时间数据
    for i in range(0, time_length):
        current_time = date_start + datetime.timedelta(hours=1 * i)
        samples.loc[i, 'time'] = current_time.strftime("%Y/%m/%d %H:%M")
        today = int(current_time.strftime("%w"))
        if 1 <= today <= 5:
            if 9 <= current_time.hour <= 17:
                samples.loc[i, 'havePeople'] = 0
            else:
                samples.loc[i, 'havePeople'] = 1
        else:
            samples.loc[i, 'havePeople'] = 1
        samples.loc[i, 'day'] = today
        samples.loc[i, 'hour'] = current_time.hour
    #print(samples)
    
    return samples

def get_illumination_data(samples, file_path, date_start, date_end, time_length):

    #  光照数据
    illumination_data = pd.read_csv(file_path)
    #print(illumination_data['illumination'])
    #print(illumination_data.loc[12, 'illumination'])

    #  滤出相应数据
    data_Len = len(illumination_data)
    for i in range(0, data_Len):
        myDataTime = datetime.datetime.strptime(illumination_data.loc[i, "time"], '%Y/%m/%d %H:%M')
        #print('myDataTime', myDataTime)
        if date_start <= myDataTime <= date_end:  #  时间范围
            timeSpan = myDataTime - date_start
            #print('timeSpan', timeSpan)
            timeID = timeSpan.days * 24 + int(timeSpan.seconds / 3600)  #时间向下取整，类似18:01的数据会视为18:00
            #print('timeID', timeID)
            samples.loc[timeID, 'illumination'] = illumination_data.loc[i, 'illumination']
        #print(samples.loc[:, ['time', 'illumination']])

    return samples

def get_meteorology_data(samples, file_path, date_start, date_end, time_length):

    #  室外数据
    meteorology_data = pd.read_csv(file_path)
    #print(meteorology_data['Temperature'])
    #  滤出相应数据
    data_Len = len(meteorology_data)
    for i in range(0, data_Len):
        myDataTime = datetime.datetime.strptime(meteorology_data.loc[i, "time"], '%Y/%m/%d %H:%M')
        if date_start <= myDataTime <= date_end:  #  时间范围
            timeSpan = myDataTime - date_start
            timeID = timeSpan.days * 24 + int(timeSpan.seconds / 3600)  #时间向下取整，类似18:01的数据会视为18:00
            samples.loc[timeID, 'outdoor_temp'] = meteorology_data.loc[i, 'Temperature']
            samples.loc[timeID, 'outdoor_pressure'] = meteorology_data.loc[i, 'Pressure']
            samples.loc[timeID, 'outdoor_humidity'] = meteorology_data.loc[i, 'Humidity']
            samples.loc[timeID, 'wind_speed'] = meteorology_data.loc[i, 'WindSpeed']
            # 风向重置编码
            #if meteorology_data.loc[i, 'WindDirection'] != Nan:
            samples.loc[timeID, 'wind_direction'] = wind_direction_code(meteorology_data.loc[i, 'WindDirection'])
            # 天气重置编码
            samples.loc[timeID, 'weather'] = Weather_code(meteorology_data.loc[i, 'Weather'])
        
    return samples

def get_heating_data(samples, file_path, date_start, date_end, time_length):

    #  供热数据
    heating_data = pd.read_csv(file_path
                               )
    #print(heating_data)

    #  滤出相应数据
    data_Len = len(heating_data)
    for i in range(0, data_Len):
        myDataTime = datetime.datetime.strptime(heating_data.loc[i, "DateTime"], '%Y/%m/%d %H:%M')
        #print('myDataTime', myDataTime)
        if date_start <= myDataTime <= date_end:  #  时间范围
            timeSpan = myDataTime - date_start
            #print('timeSpan', timeSpan)
            timeID = timeSpan.days * 24 + int(timeSpan.seconds / 3600)  #时间向下取整，类似18:01的数据会视为18:00
            #print('timeID', timeID)
            samples.loc[timeID, 'indoor_temp'] = heating_data.loc[i, '室内平均温度 (°C)']
            if '室内平均湿度 (%)' in heating_data.columns:
                samples.loc[timeID, 'indoor_humidity'] = heating_data.loc[i, '室内平均湿度 (%)']
            samples.loc[timeID, 'second_heat_temp'] = heating_data.loc[i, '二次侧供水']
            samples.loc[timeID, 'second_return_temp'] = heating_data.loc[i, '二次侧回水']
            samples.loc[timeID, 'second_heat_pressure'] = heating_data.loc[i, '二次侧供水压力']
            samples.loc[timeID, 'second_return_pressure'] = heating_data.loc[i, '二次侧回水压力']

    return samples

def drop_nan_outliers(samples):

    # 如果供温、室温为空，则删掉数据
    for i in range(0, len(samples)):
        if pd.isnull(samples.loc[i, 'indoor_temp']) and pd.isnull(samples.loc[i, 'second_heat_temp']):
            samples.drop(labels=i, axis=0, inplace=True) # 按行删除，原地替换

    samples = samples.reset_index(drop=True)  # 重设索引

    return samples

def processing_flow(save_folder_path, save_file_name, read_folder_path, illumination_file_path, meteorology_file_path, heating_file_path):

    # 读取数据
    date_start, date_end, time_length, samples = initialization()
    samples = get_time_data(samples, date_start, date_end, time_length)
    samples = get_illumination_data(samples, read_folder_path + illumination_file_path, date_start, date_end, time_length)
    samples = get_meteorology_data(samples, read_folder_path + meteorology_file_path, date_start, date_end, time_length)
    samples = get_heating_data(samples, read_folder_path + heating_file_path, date_start, date_end, time_length)
    #print(samples)

    # 去掉异常值
    ## 如果供温、室温为空，则删掉数据
    samples = drop_nan_outliers(samples)
    #print(samples)
    #print(samples.columns)
    # 保存
    samples.to_csv(save_folder_path + save_file_name + '.csv', header=True, index=False)

    # to do 构造差值时，若两个时刻不相邻，去掉相应的差值。

    return samples

if __name__ == '__main__':

    read_folder_path = 'D:/HeatingOptimization/MultipleMoments/data/preprocessing_data/'
    save_folder_path = 'D:/HeatingOptimization/MultipleMoments/result/data_preprocessing/all_sensors_average/'
    
    illumination_file_path = 'illumination/illumination.csv'
    meteorology_file_path = 'meteorology/600_center.csv'
    heating_file_path = 'heating_system/heating_system_all_sensors_average/all_dingfuhigh.csv'
    
    save_file_name = 'dingfu_high_data'

    # 数据预处理
    samples = processing_flow(save_folder_path, save_file_name, read_folder_path,
                              illumination_file_path, meteorology_file_path, heating_file_path)
    print(samples)


