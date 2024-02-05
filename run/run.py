##############################
# 选定实验模型与数据
# 重复实验取均值
##############################

#  跨文件调用
import os
import sys
current_path=os.getcwd().replace("\\","\\\\")
#current_path=os.path.dirname(os.getcwd())
sys.path.append(current_path)

from common.config import file_list, file_list_all, file_list_five, file_list_four
from processing_flow import process
from multiprocessing import Pool
from evaluation.metrics import cal_average, result_station_average, result_compare_model, result_compare_station, calculate_trustworthiness, calculate_trustworthiness_model, get_all_parameter_result
import numpy as np
from evaluation.metrics_PDP import cal_average_PDP, result_compare_station_PDP, calculate_trustworthiness_PDP, calculate_trustworthiness_model_PDP, result_compare_model_PDP

####################
# 时空可信预测
####################
if __name__ == '__main__':

    #  路径
    preparation_floder = 'D:/HeatingOptimization/STCD_2023/data/data_preparation/all_sensors_average/'
    preprocessing_floder = 'D:/HeatingOptimization/STCD_2023/result/data_preprocessing/all_sensors_average/'
    root_save_floder = 'D:/HeatingOptimization/STCD_2023/result/model_result_20230130/PDP/'
    
    #  使用的数据集
    #file_list = file_list
    file_list = file_list_four
    #file_list = file_list_all
    #file_list = ["dingfu_high", "dingfu_low", "qintaoyuan_high"]  # 不能只有一个数据集，否在生成其他站点数据时会报错
    #print(file_list)

    #  数据标准化方法
    #standard_type = 'minmax'
    standard_type = 'standard'

    #  模型类型
    model_type_list = ['Fitted_physical_model', 'Linear', 'DNN_multi_time', 'LSTM', 'ResNet', 'MDL_DNN', 'STCD_DNN_DF_TC_loss_SC', 'STCD_ResNet_all', 'STCD_MDL_all', 'STCD_LSTM_DF_TC_loss_SC', 'STCD_LSTM_DF_TC_loss', 'STCD_LSTM_DF_SC', 'STCD_LSTM_DF_TC_SC', 'STCD_LSTM_F_TC_loss_SC', 'STCD_LSTM_continuous']
    #model_type_list = ['STCD_LSTM_DF_TC_loss_SC']
    #model_type_list = ['Fitted_physical_model', 'Linear', 'STCD_LSTM_DF_TC_loss_SC']
    #model_type_list = ['Fitted_physical_model', 'Linear', 'STCD_LSTM_DF_TC_loss']
    #model_type_list = ['Linear', 'STCD_LSTM_DF_TC_loss_SC']
    #model_type_list = ['STCD_LSTM_DF_TC_loss_SC']


    #  多次重复取均值，重复次数
    repeat_time = 1
    #repeat_time = 3

    #  交叉验证
    #fold_time = 4
    fold_time = 1

    #windows_len = 6
    #windows_len_max = 13

    long_predict_len = 1

    #enhancement_times_max = 16
    #enhancement_times = 6
    
    #close_effect_rate_range = np.arange(0.90, 1.01, 0.01)
    close_effect_rate_range = range(90, 101, 1)
    #close_effect_rate = 0.1
    periodic_effect_rate_range = range(90, 101, 1)
    periodic_effect_rate = 1
    trend_effect_rate = 1
    SC_loss_weights_range = range(5, 11, 1)
    SC_loss_weights = 1
    # 数据增强相当于提高了短期分支的权重，可通过TC_loss_weights>1增加长期分支的权重
    TC_loss_weights_range = range(5, 16, 1)
    TC_loss_weights = 1

    # 创建多个进程，表示可以同时执行的进程数量。默认大小是CPU的核心数
    #p = Pool(4)
    for file_id in range(0, len(file_list)):  # 不同小区
        file_name = file_list[file_id]
        if file_name == 'qintaoyuan_low':
            windows_len = 4
            enhancement_times = 3
            for repeat_id in range(0, repeat_time):  # 多次重复取均值   # 模型训练有随机数，重复三遍；
                for fold_id in range(0, fold_time):  # 交叉验证
                    #enhancement_times = 6
                    folder_path = root_save_floder
                    close_effect_rate = 1

                    process(preparation_floder, preprocessing_floder, folder_path, file_list, file_name, repeat_id, fold_id, model_type_list, standard_type, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, trend_effect_rate, SC_loss_weights, TC_loss_weights)
                    #p.apply_async(process, args=(preparation_floder, preprocessing_floder, folder_path, file_list, file_name, repeat_id, fold_id, model_type_list, standard_type, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, trend_effect_rate, SC_loss_weights, TC_loss_weights,))

            # 计算多次重复的均值
            #cal_average(repeat_time, fold_time, root_save_floder, model_type_list, file_name)
        # 比较所有换热站结果
        #result_station_average(root_save_floder, model_type_list, file_list)
        # 如果我们用的是进程池，在调用join()之前必须要先close()，并且在close()之后不能再继续往进程池添加新的进程
        #p.close()
        # 进程池对象调用join，会等待进程池中所有的子进程结束完毕再去结束父进程
        #p.join()
    

    
    BT_model = 'Fitted_physical_model'
    #BT_model = 'Linear'
    folder_path = root_save_floder
    print('test')
    for file_id in range(0, len(file_list)):  #  不同小区
        file_name = file_list[file_id]
        if file_name == 'qintaoyuan_low':
            windows_len = 4
            # 计算多次重复的均值
            cal_average(repeat_time, fold_time, folder_path, model_type_list, file_name, windows_len)
            cal_average_PDP(repeat_time, fold_time, folder_path, model_type_list, file_name, windows_len)
            # 汇总不同站点的结果
            result_compare_station(folder_path, model_type_list, file_name)
            result_compare_station_PDP(folder_path, model_type_list, file_name)
            calculate_trustworthiness(folder_path, model_type_list, file_name, BT_model)
            calculate_trustworthiness_PDP(folder_path, model_type_list, file_name, BT_model)

    # 不同站点取均值
    result_station_average(folder_path, model_type_list, file_list)
    calculate_trustworthiness_model(folder_path, model_type_list, file_list, BT_model)
    calculate_trustworthiness_model_PDP(folder_path, model_type_list, file_list, BT_model)
    # 汇总不同方法的结果
    result_compare_model(folder_path, model_type_list)
    result_compare_model_PDP(folder_path, model_type_list)


    