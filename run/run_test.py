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

from common.config import file_list, file_list_all, file_list_five
from processing_flow import process
from multiprocessing import Pool
from evaluation.metrics import cal_average, result_station_average, result_compare_model, result_compare_station, calculate_trustworthiness, calculate_trustworthiness_model, get_all_parameter_result
import numpy as np

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
    preparation_floder = 'D:/HeatingOptimization/STCDRank/data/data_preparation/all_sensors_average/'
    preprocessing_floder = 'D:/HeatingOptimization/STCDRank/result/data_preprocessing/all_sensors_average/'
    root_save_floder = 'D:/HeatingOptimization/STCDRank/result/result_20221111/local_test/add_haihe_high/'
    
    #  使用的数据集
    #file_list = file_list
    file_list = file_list_five
    #file_list = file_list_all
    #file_list = ["dingfu_high", "dingfu_low", "qintaoyuan_high"]  # 不能只有一个数据集，否在生成其他站点数据时会报错：ValueError: a must be greater than 0 unless no samples are taken
    #file_list = ["qintaoyuan_low"]  # 不能只有一个数据集，否在生成其他站点数据时会报错：ValueError: a must be greater than 0 unless no samples are taken
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


    #  多次重复取均值，重复次数
    repeat_time = 1
    #repeat_time = 3

    #  交叉验证
    #fold_time = 4
    fold_time = 1

    #windows_len = 6
    #windows_len_max = 13

    long_predict_len_range = range(1, 6, 1)
    #long_predict_len = 1

    enhancement_times_max = 21
    #enhancement_times = 1
    
    close_effect_rate_range = np.arange(0.995, 1.0005, 0.0005)
    close_effect_rate_range = np.around(close_effect_rate_range, 4)
    #close_effect_rate = 1
    periodic_effect_rate_range = np.arange(0.995, 1.0005, 0.0005)
    periodic_effect_rate_range = np.around(periodic_effect_rate_range, 4)
    #periodic_effect_rate = 1
    SC_loss_weights_range = np.arange(0.5, 1.55, 0.05)
    SC_loss_weights_range = np.around(SC_loss_weights_range, 2)
    #SC_loss_weights = 1
    TC_loss_weights_range = np.arange(0.5, 1.55, 0.05)
    TC_loss_weights_range = np.around(TC_loss_weights_range, 2)
    #TC_loss_weights_range = range(7, 21, 1)
    #TC_loss_weights = 1

    # 创建多个进程，表示可以同时执行的进程数量。默认大小是CPU的核心数
    
    #p = Pool(8)
    for file_id in range(0, len(file_list)):  # 不同小区
        file_name = file_list[file_id]
        if file_name == 'dingfu_high':
            windows_len = 5
            enhancement_times = 6
            TC_loss_weights = 1.05
            close_effect_rate = 0.999
            periodic_effect_rate = 1
            SC_loss_weights = 0.8
        if file_name == 'dingfu_low':
            windows_len = 2
            enhancement_times = 3
            TC_loss_weights = 1.3
            close_effect_rate = 0.9995
            periodic_effect_rate = 0.9995
            SC_loss_weights = 0.95
        if file_name == 'haihe_high':
            windows_len = 3
            enhancement_times = 6
            TC_loss_weights = 0.9
            close_effect_rate = 0.9995
            periodic_effect_rate = 0.9995
            SC_loss_weights = 1.3
        if file_name == 'qintaoyuan_high':
            windows_len = 6
            enhancement_times = 2
            TC_loss_weights = 1.05
            close_effect_rate = 0.999
            periodic_effect_rate = 0.9995
            SC_loss_weights = 0.85
        if file_name == 'qintaoyuan_low':
            windows_len = 5
            enhancement_times = 2
            TC_loss_weights = 1.15
            close_effect_rate = 0.9975
            periodic_effect_rate = 0.999
            SC_loss_weights = 1.25
        for repeat_id in range(0, repeat_time):  # 多次重复取均值   # 模型训练有随机数，重复三遍；
            for fold_id in range(0, fold_time):  # 交叉验证
                for long_predict_len in long_predict_len_range:
                    folder_path = root_save_floder + '/long_predict_len_' + str(long_predict_len) + '/'
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    #print('file_id: ', file_id)
                    #print('repeat_id:', repeat_id)
                    #print('fold_id:', fold_id)
                    #print('enhancement_times', enhancement_times)
                    process(preparation_floder, preprocessing_floder, folder_path, file_list, file_name, repeat_id, fold_id, model_type_list, standard_type, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, SC_loss_weights, TC_loss_weights)
                    #p.apply_async(process, args=(preparation_floder, preprocessing_floder, folder_path, file_list, file_name, repeat_id, fold_id, model_type_list, standard_type, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, SC_loss_weights, TC_loss_weights,))

        # 计算多次重复的均值
        #cal_average(repeat_time, fold_time, root_save_floder, model_type_list, file_name)
    # 比较所有换热站结果
    #result_station_average(root_save_floder, model_type_list, file_list)
    # 如果我们用的是进程池，在调用join()之前必须要先close()，并且在close()之后不能再继续往进程池添加新的进程
    #p.close()
    # 进程池对象调用join，会等待进程池中所有的子进程结束完毕再去结束父进程
    #p.join()
    
    '''
    file_list = ['dingfu_high']
    for windows_len in range(1, 13): # [1, 13)
        # 创建保存文件夹
        folder_path = root_save_floder + 'windows_len_' + str(windows_len) + '/'
        #####  开启并行后，计算均值
        for file_id in range(0, len(file_list)):  #  不同小区
            file_name = file_list[file_id]
            # 计算多次重复的均值
            cal_average(repeat_time, fold_time, folder_path, model_type_list, file_name, windows_len)
            # 汇总不同站点的结果
            result_compare_station(folder_path, model_type_list, file_name, windows_len)
        # 不同站点取均值
        result_station_average(folder_path, model_type_list, file_list, windows_len)
        # 汇总不同方法的结果
        result_compare_model(folder_path, model_type_list, windows_len)
    '''
    '''
    BT_model = 'Fitted_physical_model'
    #BT_model = 'Linear'
    for file_id in range(0, len(file_list)):  #  不同小区
        file_name = file_list[file_id]
        # 计算多次重复的均值
        cal_average(repeat_time, fold_time, root_save_floder, model_type_list, file_name, windows_len)
        # 汇总不同站点的结果
        result_compare_station(root_save_floder, model_type_list, file_name, windows_len)
        calculate_trustworthiness(root_save_floder, model_type_list, file_name, windows_len, BT_model)
    
    # 不同站点取均值
    result_station_average(root_save_floder, model_type_list, file_list, windows_len)
    calculate_trustworthiness_model(root_save_floder, model_type_list, file_list, windows_len, BT_model)
    # 汇总不同方法的结果
    result_compare_model(root_save_floder, model_type_list, windows_len)
    '''
    '''
    #BT_model = 'Fitted_physical_model'
    BT_model = 'Linear'
    for enhancement_times in range(1, enhancement_times_max):
        folder_path = root_save_floder + 'enhancement_' + str(enhancement_times) + '/'

        for file_id in range(0, len(file_list)):  #  不同小区
            file_name = file_list[file_id]
            if file_name == 'dingfu_high':
                windows_len = 5
            if file_name == 'dingfu_low':
                windows_len = 4
            if file_name == 'qintaoyuan_high':
                windows_len = 11
            # 计算多次重复的均值
            cal_average(repeat_time, fold_time, folder_path, model_type_list, file_name, windows_len)
            # 汇总不同站点的结果
            result_compare_station(folder_path, model_type_list, file_name)
            calculate_trustworthiness(folder_path, model_type_list, file_name, BT_model)
        
        # 不同站点取均值
        result_station_average(folder_path, model_type_list, file_list)
        calculate_trustworthiness_model(folder_path, model_type_list, file_list, BT_model)
        # 汇总不同方法的结果
        result_compare_model(folder_path, model_type_list)
        
    parameter_name = 'enhancement'
    parameter_list = range(1, enhancement_times_max)
    get_all_parameter_result(root_save_floder, BT_model, parameter_name, parameter_list)
    '''
    '''
    # 参数个性化
    #BT_model = 'Fitted_physical_model'
    BT_model = 'Linear'
    for file_id in range(0, len(file_list)):  # 不同小区
        file_name = file_list[file_id]
        for close_effect_rate in close_effect_rate_range:
            # 创建保存文件夹
            folder_path = root_save_floder + file_name + '/close_effect_rate_' + str(close_effect_rate) + '/'
            
            #for file_id in range(0, len(file_list)):  #  不同小区
            #    file_name = file_list[file_id]
            # 计算多次重复的均值
            cal_average(repeat_time, fold_time, folder_path, model_type_list, file_name, windows_len)
            # 汇总不同站点的结果
            result_compare_station(folder_path, model_type_list, file_name, windows_len)
            calculate_trustworthiness(folder_path, model_type_list, file_name, windows_len, BT_model)
        
            # 不同站点取均值
            file_list_one = [file_name]
            result_station_average(folder_path, model_type_list, file_list_one, windows_len)
            calculate_trustworthiness_model(folder_path, model_type_list, file_list_one, windows_len, BT_model)
            # 汇总不同方法的结果
            result_compare_model(folder_path, model_type_list, windows_len)
            
        parameter_name = 'close_effect_rate'
        parameter_list = close_effect_rate_range
        folder_path_personalized = root_save_floder + file_name + '/'
        get_all_parameter_result(folder_path_personalized, windows_len, BT_model, parameter_name, parameter_list)
    '''