import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from evaluation.metrics import save_repeat_fold_average, save_one_station_result, save_all_model_result
from common.config import PDP_grid_resolution


# 结果取均值
def cal_average_PDP(repeat_time, fold_time, root_save, model_type_list, file_name, windows_len):

    for model_id in range(0, len(model_type_list)):  # 不同方法
        model_type = model_type_list[model_id]

        # score 
        # 计算PDP时不用重新计算score了
        '''
        input_csv_name = '_indoor_temp_next_score.csv'
        output_csv_name = 'all_score_average_'
        loc_in_file = 1
        save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)
        '''

        # direction
        ## 先计算改变每一个t~t-change_time_len时刻特征对应的可信度
        loc_in_file = 0
        #for i in range(1, windows_len+1):
        #    change_time_len = i
        for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
            if change_time_len == 1:
                change_time_len = 1
                change_time_len_str = 'short' 
            if change_time_len == 2:
                change_time_len = windows_len
                change_time_len_str = 'long'
            ## heat_temp
            input_csv_name = '_second_heat_temp' + '_change_' + str(change_time_len) + '_PDP_indoor_temp_average.csv'
            output_csv_name = 'all_direction_average_heat_temp' + '_change_' + change_time_len_str + '_PDP' + '_'
            save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)
            ## outdoor_temp
            input_csv_name = '_outdoor_temp' + '_change_' + str(change_time_len) + '_PDP_indoor_temp_average.csv'
            output_csv_name = 'all_direction_average_outdoor_temp' + '_change_' + change_time_len_str + '_PDP' + '_'
            save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)



def result_compare_station_PDP(root_save, model_type_list, file_name):

    # score
    # 计算PDP时不用重新计算score了
    '''
    score_data = pd.DataFrame(['indoor_temp_next_mse', 'indoor_temp_next_mae', 'indoor_temp_next_mape'], columns=["name"])
    input_file_name = 'all_score_average_'
    output_file_name = 'one_station_score_'
    save_one_station_result(root_save, model_type_list, file_name, score_data, input_file_name, output_file_name)
    '''

    # direction
    ## 用任意一个模型，选Fitted_physical_model 模型的PDP_grid_value作为direction_data的横坐标。不同站点的数据不同，不能用。
    ## 用0-100作为横坐标
    range_list = range(0, PDP_grid_resolution, 1)
    #print(range_list)
    range_list = list(range_list)
    #print(range_list)
    direction_data = pd.DataFrame(range_list, columns=["diff_heat"])
    print(direction_data)
    #direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])
    
    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'
        ## heat_temp
        input_file_name = 'all_direction_average_heat_temp_change_' + change_time_len_str + '_PDP' + '_'
        output_file_name = 'one_station_direction_heat_temp_change_' + change_time_len_str + '_PDP' + '_'
        save_one_station_result(root_save, model_type_list, file_name, direction_data, input_file_name, output_file_name)
        ## outdoor_temp
        input_file_name = 'all_direction_average_outdoor_temp_change_' + change_time_len_str + '_PDP' + '_'
        output_file_name = 'one_station_direction_outdoor_temp_change_' + change_time_len_str + '_PDP' + '_'
        save_one_station_result(root_save, model_type_list, file_name, direction_data, input_file_name, output_file_name)



def calculate_trustworthiness_PDP(root_save_floder, model_type_list, file_name, BT_model):

    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        for direction_col in ['heat_temp', 'outdoor_temp']:
            read_file_path = root_save_floder + 'one_station_direction_' + direction_col + '_change_' + change_time_len_str + '_PDP' + '_' + file_name + '.csv'
            target_data = pd.read_csv(read_file_path, header=0)
            # 去掉0
            #target_data = target_data[target_data['diff_heat'] != 0]
            # 去掉无用的索引
            target_data = target_data.loc[:, model_type_list]

            average_ST = np.zeros((2, len(model_type_list)), dtype=np.float)
            #col_list = []
            # 计算可信度
            for model_id in range(0, len(model_type_list)):  # 不同方法
                model_type = model_type_list[model_id]
                col_name_ST = model_type + '_ST'
                #col_list.append(col_name_ST)
                target_data[col_name_ST] = 100 * (1 - np.abs((target_data[BT_model] - target_data[model_type])/target_data[BT_model]))
                target_data.to_csv(root_save_floder + 'ST_' + direction_col + '_change_' + change_time_len_str + '_PDP' + '_' + file_name + '.csv', header=True)
                
                # 取-p~p的均值
                #average_ST[0, model_id] = np.average(target_data.loc[10:30, col_name_ST])
                # 受限PD考虑-1~1，和-2~2的区别。PD没必要区分第0列和第1列了。
                average_ST[0, model_id] = np.average(target_data[col_name_ST])
                average_ST[1, model_id] = np.average(target_data[col_name_ST])
            
            average_ST_df = pd.DataFrame(average_ST, columns=model_type_list)
            average_ST_df.to_csv(root_save_floder + 'ST_average_' + direction_col + '_change_' + change_time_len_str + '_PDP' + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)

    return target_data



def calculate_trustworthiness_model_PDP(root_save_floder, model_type_list, file_list, BT_model):

    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        all_stations_model_ST_1_1 = pd.DataFrame(columns=model_type_list)
        all_stations_model_ST_2_2 = pd.DataFrame(columns=model_type_list)

        
        for file_id in range(0, len(file_list)):  #  不同小区
            file_name = file_list[file_id]
            
            model_ST_1_1 = pd.DataFrame(columns=model_type_list)
            model_ST_2_2 = pd.DataFrame(columns=model_type_list)
            for direction_col in ['heat_temp', 'outdoor_temp']:

                read_file_path = root_save_floder + 'ST_average_' + direction_col + '_change_' + change_time_len_str + '_PDP' + '_' + file_name + '_' + BT_model + '.csv'
                target_data = pd.read_csv(read_file_path, header=0)
                model_ST_1_1 = model_ST_1_1.append(target_data.loc[0, :], ignore_index=True)
                model_ST_2_2 = model_ST_2_2.append(target_data.loc[1, :], ignore_index=True)
                '''
                target_data['file_name'] = file_name
                all_data_1_1 = all_data_1_1.append(target_data.loc[0, :], ignore_index=True)
                all_data_2_2 = all_data_2_2.append(target_data.loc[1, :], ignore_index=True)
                '''
            # 取均值
            #print(model_ST_1_1)
            #print(model_ST_1_1.mean(axis=0))
            #print(model_ST_1_1.mean(axis=0).values)
            model_ST_1_1.loc[2, :] = model_ST_1_1.mean(axis=0).values
            model_ST_2_2.loc[2, :] = model_ST_2_2.mean(axis=0).values
            #print(model_ST_1_1)

            model_ST_1_1.to_csv(root_save_floder + 'Model_ST_1_1_change_' + change_time_len_str + '_PDP' + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)
            model_ST_2_2.to_csv(root_save_floder + 'Model_ST_2_2_change_' + change_time_len_str + '_PDP' + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)

            # 汇总结果
            all_stations_model_ST_1_1 = all_stations_model_ST_1_1.append(model_ST_1_1.loc[2, :], ignore_index=True)
            all_stations_model_ST_2_2 = all_stations_model_ST_2_2.append(model_ST_2_2.loc[2, :], ignore_index=True)

        # 取均值
        all_stations_model_ST_1_1 = all_stations_model_ST_1_1.append(all_stations_model_ST_1_1.mean(axis=0), ignore_index=True)
        all_stations_model_ST_2_2 = all_stations_model_ST_2_2.append(all_stations_model_ST_2_2.mean(axis=0), ignore_index=True)
        # 保存汇总结果
        all_stations_model_ST_1_1.to_csv(root_save_floder + 'all_stations_model_ST_1_1_change_' + change_time_len_str + '_PDP' + '_' + BT_model + '.csv', header=True)
        all_stations_model_ST_2_2.to_csv(root_save_floder + 'all_stations_model_ST_2_2_change_' + change_time_len_str + '_PDP' + '_' + BT_model + '.csv', header=True)


    return all_stations_model_ST_1_1


def result_compare_model_PDP(root_save, model_type_list):

    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        # score
        # 计算PDP时不用重新计算score了
        '''
        score_data = pd.DataFrame(['indoor_temp_next_mse', 'indoor_temp_next_mae', 'indoor_temp_next_mape'], columns=["name"])
        input_file_name = 'all_score.csv'
        output_file_name = 'all_stations_average_score.csv'
        save_all_model_result(root_save, model_type_list, score_data, input_file_name, output_file_name)
        '''

        # direction
        range_list = range(0, PDP_grid_resolution, 1)
        range_list = list(range_list)
        direction_data = pd.DataFrame(range_list, columns=["diff_heat"])
        #print(direction_data)
        #direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])

        ## heat_temp
        input_file_name = 'all_direction_heat_temp_change_' + change_time_len_str + '_PDP' + '.csv'
        output_file_name = 'all_stations_average_direction_heat_temp_change_' + change_time_len_str + '_PDP' + '.csv'
        save_all_model_result(root_save, model_type_list, direction_data, input_file_name, output_file_name)
        ## outdoor_temp
        input_file_name = 'all_direction_outdoor_temp_change_' + change_time_len_str + '_PDP' + '.csv'
        output_file_name = 'all_stations_average_direction_outdoor_temp_change_' + change_time_len_str + '_PDP' + '.csv'
        save_all_model_result(root_save, model_type_list, direction_data, input_file_name, output_file_name)

