import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


def mean_absolute_percentage_error(y_true, y_pred):
    # return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    data_len = len(y_true)
    error_sum = 0.0
    error_num = 0.0
    mape = 0.0
    for i in range(0, data_len):
        if y_true[i] != 0.0:
            error_sum = error_sum + np.abs((y_true[i] - y_pred[i]) / y_true[i])
            error_num = error_num + 1
    mape = error_sum / error_num
    mape = 100 * mape
    return mape


def calculate_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, mae, mape

def save_score_predict_indoor_next(save_path, name, predict_data, ground_data):
    mse, mae, mape = calculate_score(ground_data, predict_data)
    #  保存文件
    np.savetxt(save_path + name + '.csv',
               predict_data, fmt='%s', delimiter=',')
    #print(name + '_mae:', mae)
    #  保存score
    scoreSave = open(save_path + name + '_score.csv', 'w')
    scoreSave.writelines(name + '_mse,' + str(mse) + '\n')
    scoreSave.writelines(name + '_mae,' + str(mae) + '\n')
    scoreSave.writelines(name + '_mape,' + str(mape) + '\n')
    scoreSave.close()
    return mse, mae, mape





def save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file):

        score_data = pd.DataFrame()
        for repeat_id in range(0, repeat_time):
            for fold_id in range(0, fold_time):
                read_file_path = root_save + model_type + '/' + file_name + '_' + 'repeat_' + \
                    str(repeat_id) + '_fold_' + str(fold_id) + input_csv_name
                target_data = pd.read_csv(read_file_path, header=None)
                name = str(repeat_id) + '_' + str(fold_id)
                current_data = pd.DataFrame(
                    target_data.iloc[:, loc_in_file].values, columns=[name])
                score_data = pd.concat([score_data, current_data], axis=1)
        temp = score_data.copy()
        score_data["average"] = temp.mean(axis=1)
        score_data.to_csv(root_save + model_type + '/' + output_csv_name + file_name + '.csv', header=True)
    


# 结果取均值
def cal_average(repeat_time, fold_time, root_save, model_type_list, file_name, windows_len):

    for model_id in range(0, len(model_type_list)):  # 不同方法
        model_type = model_type_list[model_id]

        # score
        input_csv_name = '_indoor_temp_next_score.csv'
        output_csv_name = 'all_score_average_'
        loc_in_file = 1
        save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)

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
            input_csv_name = '_second_heat_temp' + '_change_' + str(change_time_len) + '_direction_diff.csv'
            output_csv_name = 'all_direction_average_heat_temp' + '_change_' + change_time_len_str + '_'
            save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)
            ## outdoor_temp
            input_csv_name = '_outdoor_temp' + '_change_' + str(change_time_len) + '_direction_diff.csv'
            output_csv_name = 'all_direction_average_outdoor_temp' + '_change_' + change_time_len_str + '_'
            save_repeat_fold_average(repeat_time, fold_time, root_save, model_type, file_name, input_csv_name, output_csv_name, loc_in_file)

        
def save_all_station_result(root_save, model_type_list, file_list, score_data, input_file_name, output_file_name):

    for model_id in range(0, len(model_type_list)):  # 方法
        model_type = model_type_list[model_id]
        root_save_path = root_save + model_type + '/'

        # 重置score_data
        score_data_header = score_data
        score_data_values = pd.DataFrame()
        for file_id in range(0, len(file_list)):  # 小区
            file_name = file_list[file_id]
            read_file_path = root_save_path + input_file_name + file_name + '.csv'
            target_data = pd.read_csv(read_file_path, header=0)
            current_data = pd.DataFrame(target_data['average'].values, columns=[
                                        model_type + '_' + file_name])
            score_data_values = pd.concat([score_data_values, current_data], axis=1)
        
        # 求不同站点的均值
        temp = score_data_values.copy()
        score_data_values["average"] = temp.mean(axis=1)

        # 增加表头
        score_data_all = pd.concat([score_data_header, score_data_values], axis=1)
        score_data_all.to_csv(root_save_path + output_file_name, header=True)


def result_station_average(root_save, model_type_list, file_list):

    # score
    score_data = pd.DataFrame(['indoor_temp_next_mse', 'indoor_temp_next_mae', 'indoor_temp_next_mape'], columns=["name"])
    input_file_name = 'all_score_average_'
    output_file_name = 'all_score.csv'
    save_all_station_result(root_save, model_type_list, file_list, score_data, input_file_name, output_file_name)

    # direction
    direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])
   
    #for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
    #    if change_time_len == 2:
    #        change_time_len = windows_len
    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'
        ## heat_temp
        input_file_name = 'all_direction_average_heat_temp_change_' + change_time_len_str + '_'
        output_file_name = 'all_direction_heat_temp_change_' + change_time_len_str + '.csv'
        save_all_station_result(root_save, model_type_list, file_list, direction_data, input_file_name, output_file_name)
        ## outdoor_temp
        input_file_name = 'all_direction_average_outdoor_temp_change_' + change_time_len_str + '_'
        output_file_name = 'all_direction_outdoor_temp_change_' + change_time_len_str + '.csv'
        save_all_station_result(root_save, model_type_list, file_list, direction_data, input_file_name, output_file_name)
    
    ## 汇总改变1~windows_len个时刻特征对应的可信度，不用计算均值
    input_file_name = 'all_direction_heat_temp_change_'
    output_file_name = 'all_direction_heat_temp_change_time_all.csv'
    save_change_time_len(root_save, model_type_list, direction_data, input_file_name, output_file_name)
    input_file_name = 'all_direction_outdoor_temp_change_'
    output_file_name = 'all_direction_outdoor_temp_change_time_all.csv'
    save_change_time_len(root_save, model_type_list, direction_data, input_file_name, output_file_name)


def save_change_time_len(root_save, model_type_list, score_data, input_file_name, output_file_name):

    for model_id in range(0, len(model_type_list)):  # 方法
        model_type = model_type_list[model_id]
        root_save_path = root_save + model_type + '/'

        #for change_time_len in range(1, windows_len+1):
        #for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        #    if change_time_len == 2:
        #        change_time_len = windows_len
        for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
            if change_time_len == 1:
                change_time_len_str = 'short' 
            if change_time_len == 2:
                change_time_len_str = 'long'

            input_file_name_current = input_file_name + change_time_len_str + '.csv'
            #print('input_file_name', input_file_name_current)
            read_file_path = root_save_path + input_file_name_current
            #print('read_file_path', read_file_path)
            target_data = pd.read_csv(read_file_path, header=0)
            current_data = pd.DataFrame(target_data['average'].values, columns=['change_time_len_' + change_time_len_str])
            score_data = pd.concat([score_data, current_data], axis=1)

        score_data.to_csv(root_save_path + output_file_name, header=True)


def result_compare_model(root_save, model_type_list):

    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        # score
        score_data = pd.DataFrame(['indoor_temp_next_mse', 'indoor_temp_next_mae', 'indoor_temp_next_mape'], columns=["name"])
        input_file_name = 'all_score.csv'
        output_file_name = 'all_stations_average_score.csv'
        save_all_model_result(root_save, model_type_list, score_data, input_file_name, output_file_name)

        # direction
        direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])

        ## heat_temp
        input_file_name = 'all_direction_heat_temp_change_' + change_time_len_str + '.csv'
        output_file_name = 'all_stations_average_direction_heat_temp_change_' + change_time_len_str + '.csv'
        save_all_model_result(root_save, model_type_list, direction_data, input_file_name, output_file_name)
        ## outdoor_temp
        input_file_name = 'all_direction_outdoor_temp_change_' + change_time_len_str + '.csv'
        output_file_name = 'all_stations_average_direction_outdoor_temp_change_' + change_time_len_str + '.csv'
        save_all_model_result(root_save, model_type_list, direction_data, input_file_name, output_file_name)


def save_all_model_result(root_save, model_type_list, score_data, input_file_name, output_file_name):

    # 
    for model_id in range(0, len(model_type_list)):  # 方法
        # 读取数据路径
        model_type = model_type_list[model_id]
        root_save_path = root_save + model_type + '/'
        read_file_path = root_save_path + input_file_name

        target_data = pd.read_csv(read_file_path, header=0)
        current_data = pd.DataFrame(target_data['average'].values, columns=[model_type])
        score_data = pd.concat([score_data, current_data], axis=1)

    score_data.to_csv(root_save + output_file_name, header=True)



def result_compare_station(root_save, model_type_list, file_name):

    # score
    score_data = pd.DataFrame(['indoor_temp_next_mse', 'indoor_temp_next_mae', 'indoor_temp_next_mape'], columns=["name"])
    input_file_name = 'all_score_average_'
    output_file_name = 'one_station_score_'
    save_one_station_result(root_save, model_type_list, file_name, score_data, input_file_name, output_file_name)

    # direction
    direction_data = pd.DataFrame([-2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0], columns=["diff_heat"])
    
    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'
        ## heat_temp
        input_file_name = 'all_direction_average_heat_temp_change_' + change_time_len_str + '_'
        output_file_name = 'one_station_direction_heat_temp_change_' + change_time_len_str + '_'
        save_one_station_result(root_save, model_type_list, file_name, direction_data, input_file_name, output_file_name)
        ## outdoor_temp
        input_file_name = 'all_direction_average_outdoor_temp_change_' + change_time_len_str + '_'
        output_file_name = 'one_station_direction_outdoor_temp_change_' + change_time_len_str + '_'
        save_one_station_result(root_save, model_type_list, file_name, direction_data, input_file_name, output_file_name)



def save_one_station_result(root_save, model_type_list, file_name, score_data, input_file_name, output_file_name):

    for model_id in range(0, len(model_type_list)):  # 方法
        model_type = model_type_list[model_id]
        root_save_path = root_save + model_type + '/'


        read_file_path = root_save_path + input_file_name + file_name + '.csv'
        target_data = pd.read_csv(read_file_path, header=0)
        current_data = pd.DataFrame(target_data['average'].values, columns=[
                                    model_type])
        score_data = pd.concat([score_data, current_data], axis=1)
        

    # 增加表头
    # 保存到最外层文件夹
    score_data.to_csv(root_save + output_file_name + file_name + '.csv', header=True)


def calculate_trustworthiness(root_save_floder, model_type_list, file_name, BT_model):

    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        for direction_col in ['heat_temp', 'outdoor_temp']:
            read_file_path = root_save_floder + 'one_station_direction_' + direction_col + '_change_' + change_time_len_str + '_' + file_name + '.csv'
            target_data = pd.read_csv(read_file_path, header=0)
            # 去掉0
            target_data = target_data[target_data['diff_heat'] != 0]
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
                target_data.to_csv(root_save_floder + 'ST_' + direction_col + '_change_' + change_time_len_str + '_' + file_name + '.csv', header=True)
                
                # 取-p~p的均值
                average_ST[0, model_id] = np.average(target_data.loc[10:30, col_name_ST])
                average_ST[1, model_id] = np.average(target_data[col_name_ST])
            
            average_ST_df = pd.DataFrame(average_ST, columns=model_type_list)
            average_ST_df.to_csv(root_save_floder + 'ST_average_' + direction_col + '_change_' + change_time_len_str + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)

    return target_data



def calculate_trustworthiness_model(root_save_floder, model_type_list, file_list, BT_model):

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

                read_file_path = root_save_floder + 'ST_average_' + direction_col + '_change_' + change_time_len_str + '_' + file_name + '_' + BT_model + '.csv'
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

            model_ST_1_1.to_csv(root_save_floder + 'Model_ST_1_1_change_' + change_time_len_str + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)
            model_ST_2_2.to_csv(root_save_floder + 'Model_ST_2_2_change_' + change_time_len_str + '_' + file_name + '_' + BT_model + '.csv', header=True, index=None)

            # 汇总结果
            all_stations_model_ST_1_1 = all_stations_model_ST_1_1.append(model_ST_1_1.loc[2, :], ignore_index=True)
            all_stations_model_ST_2_2 = all_stations_model_ST_2_2.append(model_ST_2_2.loc[2, :], ignore_index=True)

        # 取均值
        all_stations_model_ST_1_1 = all_stations_model_ST_1_1.append(all_stations_model_ST_1_1.mean(axis=0), ignore_index=True)
        all_stations_model_ST_2_2 = all_stations_model_ST_2_2.append(all_stations_model_ST_2_2.mean(axis=0), ignore_index=True)
        # 保存汇总结果
        all_stations_model_ST_1_1.to_csv(root_save_floder + 'all_stations_model_ST_1_1_change_' + change_time_len_str + '_' + BT_model + '.csv', header=True)
        all_stations_model_ST_2_2.to_csv(root_save_floder + 'all_stations_model_ST_2_2_change_' + change_time_len_str + '_' + BT_model + '.csv', header=True)


    return all_stations_model_ST_1_1

def get_all_parameter_result(root_save_floder, BT_model, parameter_name, parameter_list):


    for change_time_len in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
        if change_time_len == 1:
            change_time_len_str = 'short' 
        if change_time_len == 2:
            change_time_len_str = 'long'

        all_parameter_score = pd.DataFrame()
        all_parameter_trustworthiness_1_1 = pd.DataFrame()
        all_parameter_trustworthiness_2_2 = pd.DataFrame()
        
        for parameter_id in parameter_list:
            folder_path = root_save_floder + parameter_name + '_' + str(parameter_id) + '/'

            #if parameter_name == 'windows_len':
            #    windows_len = parameter_id

            # 汇总预测精度
            parameter_score_data = pd.read_csv(folder_path + 'all_stations_average_score.csv', header=0)
            parameter_score_data[parameter_name] = parameter_id

            all_parameter_score = all_parameter_score.append(parameter_score_data, ignore_index=True)

            # 汇总预测可信度
            trustworthiness_1_1 = pd.read_csv(folder_path + 'all_stations_model_ST_1_1_change_' + change_time_len_str + '_' + BT_model + '.csv', header=0)
            trustworthiness_1_1[parameter_name] = parameter_id
            all_parameter_trustworthiness_1_1 = all_parameter_trustworthiness_1_1.append(trustworthiness_1_1, ignore_index=True)
            
            trustworthiness_2_2 = pd.read_csv(folder_path + 'all_stations_model_ST_2_2_change_' + change_time_len_str + '_' + BT_model + '.csv', header=0)
            trustworthiness_2_2[parameter_name] = parameter_id
            all_parameter_trustworthiness_2_2 = all_parameter_trustworthiness_2_2.append(trustworthiness_2_2, ignore_index=True)

        # 保存结果
        all_parameter_score.to_csv(root_save_floder + 'all_parameter_score_' + parameter_name + '_' + BT_model + '_change_' + change_time_len_str + '.csv', header=True, index=None)
        all_parameter_trustworthiness_1_1.to_csv(root_save_floder + 'all_parameter_trustworthiness_1_1_' + BT_model + '_change_' + change_time_len_str + '.csv', header=True, index=None)
        all_parameter_trustworthiness_2_2.to_csv(root_save_floder + 'all_parameter_trustworthiness_2_2_' + BT_model + '_change_' + change_time_len_str + '.csv', header=True, index=None)


    return folder_path


