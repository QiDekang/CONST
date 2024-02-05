#  跨文件夹调用
import numpy as np
from data_preprocessing.standard import num_standard_feature_transform
from common.config import baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols
from data_preprocessing.standard import get_direction_standard, num_standard_feature_transform_drop
from common.config import baseline_numeric_cols, baseline_std_cols
from evaluation.accuracy_baselines import model_predict_baselines
#from data_preprocessing.processing_flow import get_multi_time_data
from data_preprocessing.flow import get_multi_time_data
from evaluation.accuracy_MDL import model_predict_MDL
from common.config import PDP_grid_resolution


def model_trustworthiness_all_MDL_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, test_t_0_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other):

    direction_col_name_list = ['second_heat_temp', 'outdoor_temp']
    for i in range(0, len(direction_col_name_list)):
        direction_col_name = direction_col_name_list[i]
        print(direction_col_name)
        model_trustworthiness_MDL(direction_col_name, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, test_t_0_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)
    
    return 'sucess'


def model_trustworthiness_MDL(direction_col_name, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, test_t_0_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other):

    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_test_t_0_data
    
    # 等间隔数据
    max_value = test_t_0_data[direction_col_name].max()
    min_value = test_t_0_data[direction_col_name].min()
    #print(min_value, max_value)
    grid_value = np.linspace(min_value, max_value, PDP_grid_resolution)
    grid_value = np.around(grid_value, 3)

    #结果存储数组
    # PDP_grid_resolution个数据点
    time_length = np.size(continuous_model_data, 0)
    heat_temp_new_array = np.zeros((time_length, PDP_grid_resolution), dtype=np.float)
    predict_indoor_temp_array = np.zeros((time_length, PDP_grid_resolution), dtype=np.float)

    for j in np.arange(0, PDP_grid_resolution):
        #  构造新的特征
        #direction_diff = round(0.1 * (j - 20), 1)  # 供温上下调整2度, 步长0.1度
        direction_diff = grid_value[j]
        predict_indoor_temp, heat_temp_new = get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, test_t_0_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)

        #  保存数据
        heat_temp_new_array[:, j] = heat_temp_new
        #print('heat_temp_new', heat_temp_new)

        predict_indoor_temp_array[:, j] = predict_indoor_temp
    # average
    indoor_temp_average = np.average(predict_indoor_temp_array, axis=0)

    if change_time_type == 1:
        change_time_len = change_time_type
    else:
        change_time_len = windows_len

    np.savetxt(save_folder + direction_col_name + '_change_' + str(change_time_len) + '_PDP_indoor_temp_average.csv', indoor_temp_average, fmt='%s', delimiter=',')
    np.savetxt(save_folder + direction_col_name + '_change_' + str(change_time_len) + '_PDP_grid_value.csv', grid_value, fmt='%s', delimiter=',')

    return indoor_temp_average


def get_indoor_temp(direction_col_name, direction_diff, save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, test_t_0_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other):

    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_test_t_0_data

    # 需要将test_t_0_data中供温、外温增加不同大小，再重新标准化，重新构造multi_time数据

    ## 供温、外温增加不同大小
    test_t_0_data_new = test_t_0_data.copy(deep=True)
    #print('test_t_0_data_new\n', test_t_0_data_new)

    #### 供温上下调整2度, 步长0.1度
    #print('direction_diff\n', direction_diff)
    #test_t_0_data_new[direction_col_name] = test_t_0_data[direction_col_name] + direction_diff
    test_t_0_data_new[direction_col_name] = direction_diff # 直接将grid_value[j]赋值给新的数据。
    #print('test_t_0_data[direction_col_name] \n', test_t_0_data[direction_col_name])
    #print('test_t_0_data_new[direction_col_name] + diff \n', test_t_0_data_new[direction_col_name])
    #print('test_t_0_data_new \n', test_t_0_data_new)

    ##  重新标准化
    test_t_0_data_new_std = num_standard_feature_transform_drop(test_t_0_data_new, feature_ss, baseline_numeric_cols, baseline_std_cols)
    
    ##  重新构造multi_time数据
    ### 全部时刻特征均改变的数据
    multi_time_test_label_new, multi_time_test_t_0_data_new = get_multi_time_data(test_t_0_data_new_std, windows_len)
    
    ## 将重新构造的数据中change_time_len长度的数据替换原有未改变的数据
    ### 无论change_time_len等于多少，t-1数据均不用改变。若改变则差值X(t)-X(t-1)不会变，模型结果变化很小。t-1数据不改变也是有道理的，相当于每一个个时刻的供温都比不做改变时增加1度，看室温的变化。
    ### 因为Y(t)是t-1均不变时获得的真实值，我们预测Y(t+1)-Y(t)时，X(t-1)自然应该是之前没有改变的数据。
    
    

    if change_time_type == 1:
        # 替换t-0时刻
        multi_time_test_t_0_data_new = replace_some_time_data_baselines(multi_time_test_t_0_data, multi_time_test_t_0_data_new, change_time_type, windows_len)
        # 否则不处理，也就是全部替换


    ## 用预测模型预测室温
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data_new, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)


    return indoor_temp_next_predict_truth, test_t_0_data_new[direction_col_name]


def replace_some_time_data_baselines(multi_time_test_t_0_data, multi_time_test_t_0_data_new, change_time_type, windows_len):

    # 打开封装
    continuous_model_data_old, continuous_baseline_data_old, wind_data_old, weather_data_old, day_data_old, hour_data_old, havePeople_data_old = multi_time_test_t_0_data
    continuous_model_data_new, continuous_baseline_data_new, wind_data_new, weather_data_new, day_data_new, hour_data_new, havePeople_data_new = multi_time_test_t_0_data_new

    continuous_model_data_part_new = continuous_model_data_old
    continuous_baseline_data_part_new = continuous_baseline_data_old

    continuous_model_data_part_new[:, windows_len-change_time_type:windows_len, :] = continuous_model_data_new[:, windows_len-change_time_type:windows_len, :]
    continuous_baseline_data_part_new[:, windows_len-change_time_type:windows_len, :] = continuous_baseline_data_new[:, windows_len-change_time_type:windows_len, :]


    # 重新封装
    multi_time_test_t_0_part_new_data = [continuous_model_data_part_new, continuous_baseline_data_part_new, wind_data_old, weather_data_old, day_data_old, hour_data_old, havePeople_data_old]

    return multi_time_test_t_0_part_new_data