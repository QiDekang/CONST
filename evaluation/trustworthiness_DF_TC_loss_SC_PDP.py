#  跨文件夹调用
import numpy as np
from data_preprocessing.standard import num_standard_feature_transform
from data_preprocessing.label_feature import get_difference_data
from common.config import baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols
from data_preprocessing.standard import get_direction_standard
from evaluation.accuracy_DF_TC_loss_SC import model_predict_DF_TC_loss_SC
from data_preprocessing.flow import get_multi_time_data, get_long_short_data, get_all_diff_data, get_diff_std, get_diff_std_test
from data_preprocessing.label_feature import replace_some_time_data
from common.config import PDP_grid_resolution


def trustworthiness_all_DF_TC_loss_SC_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other):

    direction_col_name_list = ['second_heat_temp', 'outdoor_temp']
    for i in range(0, len(direction_col_name_list)):
        direction_col_name = direction_col_name_list[i]
        print(direction_col_name)
        trustworthiness_DF_TC_loss_SC(direction_col_name, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other)
    return 'sucess'


def trustworthiness_DF_TC_loss_SC(direction_col_name, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other):

    # 等间隔数据
    max_value = test_data[direction_col_name].max()
    min_value = test_data[direction_col_name].min()
    #print(min_value, max_value)
    grid_value = np.linspace(min_value, max_value, PDP_grid_resolution)
    grid_value = np.around(grid_value, 3)


    #结果存储数组
    # PDP_grid_resolution个数据点
    time_length = np.size(test_data, 0)
    heat_temp_new_array = np.zeros((time_length, PDP_grid_resolution), dtype=np.float)
    predict_indoor_temp_array = np.zeros((time_length, PDP_grid_resolution), dtype=np.float)

    # new
    for j in np.arange(0, PDP_grid_resolution):
        #  构造新的特征
        #direction_diff = round(0.1 * (j - 20), 1)  # 供温上下调整2度, 步长0.1度
        direction_diff = grid_value[j]
        #print('grid_value', direction_diff)
        predict_indoor_temp, heat_temp_new = get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other)
        #print('predict_indoor_temp', predict_indoor_temp)
        #  保存数据
        heat_temp_new_array[:, j] = heat_temp_new
        #print('heat_temp_new', heat_temp_new)
        predict_indoor_temp_array[:, j] = predict_indoor_temp
    # average
    indoor_temp_average = np.average(predict_indoor_temp_array, axis=0)
    #print(indoor_temp_average)

    if change_time_type == 1:
        change_time_len = change_time_type
    else:
        change_time_len = windows_len

    #np.savetxt(save_path + direction_col_name + '_new_array.csv', heat_temp_new_array, fmt='%s', delimiter=',')
    #np.savetxt(save_path + direction_col_name + '_predict_indoor_temp_direction_diff_array.csv', predict_indoor_temp_direction_diff_array, fmt='%s', delimiter=',')
    #np.savetxt(save_path + direction_col_name + 'predict_indoor_temp_array.csv', predict_indoor_temp_array, fmt='%s', delimiter=',')
    #np.savetxt(save_folder + direction_col_name + '_indoor_temp_direction_diff_average.csv', indoor_temp_direction_diff_average, fmt='%s', delimiter=',')
    np.savetxt(save_folder + direction_col_name + '_change_' + str(change_time_len) + '_PDP_indoor_temp_average.csv', indoor_temp_average, fmt='%s', delimiter=',')
    np.savetxt(save_folder + direction_col_name + '_change_' + str(change_time_len) + '_PDP_grid_value.csv', grid_value, fmt='%s', delimiter=',')

    return indoor_temp_average


def get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other):

    # 供温、外温增加不同大小
    test_data_new = test_data.copy(deep=True)
    #test_data_new[direction_col_name] = test_data[direction_col_name] + direction_diff
    test_data_new[direction_col_name] = direction_diff # 直接将grid_value[j]赋值给新的数据。

    # 重新构造多时刻数据
    multi_time_test_t_0_label_new, multi_time_test_t_0_data_new = get_multi_time_data(test_data_new, windows_len)

    # 重新构造长期差分和短期差分数据
    ## t_1和t_n数据不变，仅t_0数据改变
    multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data = multi_time_0_1_n_data
    short_label_test_new, short_data_test_new, long_label_test_new, long_data_test_new = get_all_diff_data(multi_time_test_t_0_label_new, multi_time_test_t_0_data_new, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data)
    
    # 重新标准化
    ## 所有时刻的数据均改变
    short_data_std_test_new, long_data_std_test_new = get_diff_std_test(short_data_test_new, long_data_test_new, windows_len, feature_diff_ss)
    
    # 仅改变部分时刻的数据
    # 将change_time_type长度的数据替换为新数据
    ## 先测试全部替换
    if change_time_type == 1:
        # 替换t-0时刻
        short_data_std_test_new, long_data_std_test_new = replace_some_time_data(short_data_std_test_new, long_data_std_test_new, short_data_std_test, long_data_std_test, change_time_type, windows_len)
        # 否在不处理，也就是全部替换

    # 预测
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_DF_TC_loss_SC(model_type, predict_model, label_next_ss, label_diff_ss, short_label_test_new, short_data_std_test_new, long_data_std_test_new, temporal_discount_rate_test, accuracy_data_other)


    return indoor_temp_next_predict_truth, test_data_new[direction_col_name]
