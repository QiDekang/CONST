#  跨文件夹调用
import numpy as np
from data_preprocessing.standard import num_standard_feature_transform
from common.config import baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols
from data_preprocessing.standard import get_direction_standard, num_standard_feature_transform_drop
from common.config import baseline_numeric_cols, baseline_std_cols
from evaluation.accuracy_physical import model_predict_physical
#from data_preprocessing.processing_flow import get_multi_time_data
from data_preprocessing.flow import get_multi_time_data


def model_trustworthiness_all_physical(save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted, change_time_type, windows_len):

    direction_col_name_list = ['second_heat_temp', 'outdoor_temp']
    for i in range(0, len(direction_col_name_list)):
        direction_col_name = direction_col_name_list[i]
        #print(direction_col_name)
        model_trustworthiness_physical(direction_col_name, save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted, change_time_type, windows_len)
    
    return 'sucess'


def model_trustworthiness_physical(direction_col_name, save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted, change_time_type, windows_len):

    time_length = np.size(test_t_0_data, 0)
    # -2度至2度间有41个数据点
    heat_temp_new_array = np.zeros((time_length, 41), dtype=np.float)
    predict_indoor_temp_direction_diff_array = np.zeros(
        (time_length, 41), dtype=np.float)
    predict_indoor_temp_array = np.zeros((time_length, 41), dtype=np.float)
    # first, 供温不变时预测的室温
    direction_diff = 0.0
    predict_indoor_temp_first, heat_temp_new_first = get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted)
    # new
    #print('direction')
    for j in np.arange(0, 41):
        #  构造新的特征
        direction_diff = round(0.1 * (j - 20), 1)  # 供温上下调整2度, 步长0.1度
        predict_indoor_temp, heat_temp_new = get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted)
        predict_indoor_temp_direction_diff = predict_indoor_temp - predict_indoor_temp_first
        #  保存数据
        heat_temp_new_array[:, j] = heat_temp_new
        #print('heat_temp_new', heat_temp_new)
        predict_indoor_temp_direction_diff_array[:, j] = predict_indoor_temp_direction_diff
        predict_indoor_temp_array[:, j] = predict_indoor_temp
    # average
    indoor_temp_direction_diff_average = np.average(predict_indoor_temp_direction_diff_array, axis=0)

    if change_time_type == 1:
        change_time_len = change_time_type
    else:
        change_time_len = windows_len

    #np.savetxt(save_folder + direction_col_name + '_new_array.csv', heat_temp_new_array, fmt='%s', delimiter=',')
    #np.savetxt(save_folder + direction_col_name + '_predict_indoor_temp_direction_diff_array.csv', predict_indoor_temp_direction_diff_array, fmt='%s', delimiter=',')
    #np.savetxt(save_folder + direction_col_name + '_predict_indoor_temp_array.csv', predict_indoor_temp_array, fmt='%s', delimiter=',')
    np.savetxt(save_folder + direction_col_name + '_change_' + str(change_time_len) + '_direction_diff.csv', indoor_temp_direction_diff_average, fmt='%s', delimiter=',')
    #np.savetxt(save_folder + direction_col_name + '_direction_diff.csv', indoor_temp_direction_diff_average, fmt='%s', delimiter=',')

    return indoor_temp_direction_diff_average


def get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, label_next_ss, test_t_0_data, file_name, is_fitted):

    ## 供温、外温增加不同大小
    test_t_0_data_new = test_t_0_data.copy(deep=True)
    #print('test_t_0_data_new\n', test_t_0_data_new)

    #### 供温上下调整2度, 步长0.1度
    #print('direction_diff\n', direction_diff)
    test_t_0_data_new[direction_col_name] = test_t_0_data[direction_col_name] + direction_diff


    ## 用预测模型预测室温
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_physical(model_type, label_next_ss, test_t_0_data_new, file_name, is_fitted)


    return indoor_temp_next_predict_truth, test_t_0_data_new[direction_col_name]

    