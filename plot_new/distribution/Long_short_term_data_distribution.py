#  跨文件调用
import os
import sys
current_path=os.getcwd().replace("\\","\\\\")
#current_path=os.path.dirname(os.getcwd())
sys.path.append(current_path)
##############################
# 选定实验模型与数据
# 重复实验取均值
##############################
from common.config import file_list_all, file_list_five, file_list_four
from multiprocessing import Pool
from evaluation.metrics import cal_average, result_compare_station
from data_preprocessing.processing_flow import data_preprocessing
import seaborn as sns

from plot_data_preprocessing import plot_data_preprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
####################
# 验证的部分结论
# 不能使用Y(t+n)，方向错误，起作用的是建模排序关系
# 不能使用全部特征，会使供温等主要特征的影响减弱
# 不能使用构造的特征，会使方向错误

## 时空，空间部分，可以综合三个小区预测剩下的小区。元学习。
# 时空可信预测

####################
if __name__ == '__main__':

    #  路径
    #preparation_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preparation/original_indoor_temp/'
    #preprocessing_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preprocessing/original_indoor_temp/'
    preparation_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preparation/all_sensors_average/'
    preprocessing_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preprocessing/all_sensors_average/'
    root_save_floder = 'D:/HeatingOptimization/MultipleMoments/result/model_result_20220519/all/'

    #  使用的数据集
    file_list = file_list_four
    #file_list = ['dingfu_high']  # 不能只有一个数据集，否在生成其他站点数据时会报错：ValueError: a must be greater than 0 unless no samples are taken
    #print(file_list)

    #  数据标准化方法
    #standard_type = 'minmax'
    standard_type = 'standard'

    #  模型类型
    #model_type_list = ['STCF_MFF_D']
    model_type_list = ['STCF_MFF_D', 'STCF_MFF_D_TC_MMF', 'STCF_all', 'linear', 'DNN', 'DNN_with_embedding', 'LSTM']
    #model_type_list = ['STCF_MFF']
    #model_type_list = ['STCF_MFF', 'STCF_MFF_D', 'STCF_MFF_D_TC', 'STCF_MFF_D_TC_MMF', 'STCF_all', 'linear', 'DNN', 'DNN_with_embedding', 'LSTM']
    #model_type_list = ['spatial_consistent', 'long_short_volatility']
    #model_type_list = ['long_short_volatility']

    #  多次重复取均值，重复次数
    repeat_time = 1
    #repeat_time = 1

    #  交叉验证
    fold_time = 1
    #fold_time = 1

    file_name = file_list[0]
    fold_id = 0
    train_short_volatility, train_long_volatility, test_short_volatility, test_long_volatility = plot_data_preprocessing(preparation_floder, preprocessing_floder, file_name, fold_id, standard_type)
    print(train_short_volatility)

    #plot_data = train_short_volatility.append(test_short_volatility, ignore_index=True)
    plot_data_short = train_short_volatility.append(test_short_volatility, ignore_index=True)
    plot_data_long = train_long_volatility.append(test_long_volatility, ignore_index=True)


    #mpl.rcParams['font.family'] = 'KaiTi'
    mpl.rcParams['font.family'] = 'Times New Roman'
    #mpl.rcParams['font.size'] = '15'
    plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
    #fontdict={"family": "KaiTi", "size": 15}

    #创建图形
    plt.figure(figsize=(16, 6))
    #plt.figure()
    #第一行第一列图形
    ax1 = plt.subplot(1,2,1)
    #第一行第二列图形
    ax2 = plt.subplot(1,2,2)

    #选择ax1
    plt.sca(ax1)

    sns.set(style="ticks")
    #plt.figure(figsize=(8,6))
    sns.set_context('notebook', font_scale=1.5)
    #ax = sns.scatterplot(x="HeatT", y="IndoorT", hue="outdoorT_d_3", style="outdoorT_d_3", hue_order=["level_1", "level_2", "level_3"], style_order=["level_1", "level_2", "level_3"], markers=["s", "X", "o"])
    ax = sns.scatterplot(x=plot_data_short.loc[:, "second_heat_temp_diff"], y=plot_data_short.loc[:, "indoor_temp_diff"])
    ax.set_xlabel("Heat Temperature Difference", fontsize = 28)
    ax.set_ylabel("Indoor Temperature Difference", fontsize = 28)
    ax.set_title("(a) Short-term distribution of data", fontsize = 32)
    #plt.legend(loc = 'lower left', prop = {'size': 13})
    plt.xlim((-2, 2))
    plt.ylim((-0.5, 0.5))
    plt.tick_params(labelsize=24)
    plt.subplots_adjust(bottom=0.15, left=0.15)


    #选择ax1
    plt.sca(ax2)
    sns.set(style="ticks")
    #plt.figure(figsize=(8,6))
    sns.set_context('notebook', font_scale=1.5)
    #ax = sns.scatterplot(x="HeatT", y="IndoorT", hue="outdoorT_d_3", style="outdoorT_d_3", hue_order=["level_1", "level_2", "level_3"], style_order=["level_1", "level_2", "level_3"], markers=["s", "X", "o"])
    ax = sns.scatterplot(x=plot_data_long.loc[:, "second_heat_temp_diff"], y=plot_data_long.loc[:, "indoor_temp_diff"])
    ax.set_xlabel("Heat Temperature Difference", fontsize = 28)
    ax.set_ylabel("Indoor Temperature Difference", fontsize = 28)
    ax.set_title("(b) Long-term distribution of data", fontsize = 32)
    #plt.legend(loc = 'lower left', prop = {'size': 13})
    plt.xlim((-18, 18))
    plt.ylim((-6, 6))
    plt.tick_params(labelsize=24)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
    plt.show()
