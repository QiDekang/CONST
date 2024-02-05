import os
import csv
import datetime
import numpy as np
import pandas as pd
from data_preparation_original import processing_flow

# 共用的folder
read_folder_path = 'D:/HeatingOptimization/MultipleMoments/data/preprocessing_data/'
save_folder_path = 'D:/HeatingOptimization/MultipleMoments/result/data_preparation/original_indoor_temp/'

# 共用的数据
# 仅有滨海新区一个换热站的光照数据，其他换热站缺失数据用此代替
illumination_file_path = 'illumination/illumination.csv'

'''
# 各供热站对应的数据
## dingfu_high
### 鼎福位于南开区，为天津的中心区
meteorology_file_path = 'meteorology/600_center.csv'
heating_file_path = 'heating_system/heating_system_all_sensors_average/all_dingfuhigh.csv'
save_file_name = 'dingfu_high_data'
### 数据预处理
dingfu_high_data = processing_flow(save_folder_path, save_file_name, read_folder_path,
                                   illumination_file_path, meteorology_file_path, heating_file_path)
print(dingfu_high_data)
'''


## 鼎福高、低区位于南开区，为中心区，编号600；
## 海河大观高、中、低区位于河西区，为中心区，编号600；
## 航大位于东丽区，编号610；
## 琴涛苑高、低区位于滨海新区中的塘沽区，编号615；
## 张家窝位于西青区、编号602；
## 钻石山高、中、低区位于南开区，为中心区，编号600；
meteorology_file_list = ['600_center.csv',
                         '600_center.csv',
                         '600_center.csv',
                         '600_center.csv',
                         '600_center.csv',
                         '610.csv',
                         '615.csv',
                         '615.csv',
                         '602.csv',
                         '600_center.csv',
                         '600_center.csv',
                         '600_center.csv']

heating_file_list = ['dingfu_high.csv',
                     'dingfu_low.csv',
                     'haihe_high.csv',
                     'haihe_low.csv',
                     'haihe_mid.csv',
                     'hangda.csv',
                     'qintaoyuan_high.csv',
                     'qintaoyuan_low.csv',
                     'zhangjiawo.csv',
                     'zuanshishan_high.csv',
                     'zuanshishan_low.csv',
                     'zuanshishan_mid.csv']

save_file_name_list = ['dingfu_high_data',
                       'dingfu_low_data', 
                       'haihe_high_data', 
                       'haihe_low_data',
                       'haihe_mid_data',
                       'hangda_data',
                       'qintaoyuan_high_data',
                       'qintaoyuan_low_data',
                       'zhangjiawo_data',
                       'zuanshishan_high_data',
                       'zuanshishan_low_data',
                       'zuanshishan_mid_data']

for i in range(0, len(heating_file_list)):
    meteorology_file_path = 'meteorology/' + meteorology_file_list[i]
    heating_file_path = 'heating_system/original_indoor_temp/' + heating_file_list[i]
    save_file_name = save_file_name_list[i]
    print(save_file_name)
    ### 数据预处理
    samples = processing_flow(save_folder_path, save_file_name, read_folder_path,
                                    illumination_file_path, meteorology_file_path, heating_file_path)
    #print(samples)