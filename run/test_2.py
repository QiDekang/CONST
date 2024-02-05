import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


def calculate_trustworthiness(root_save_floder, model_type_list, file_name, windows_len):

    for direction_col in ['heat_temp']:
        read_file_path = root_save_floder + 'one_station_direction_' + direction_col + '_change_' + str(windows_len) + '_' + file_name + '.csv'
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
            target_data[col_name_ST] = 100 * (1 - np.abs((target_data['fitted_physical_model'] - target_data[model_type])/target_data['fitted_physical_model']))
            target_data.to_csv(root_save_floder + 'ST_' + direction_col + '_change_' + str(windows_len) + '_' + file_name + '.csv', header=True)
            
            # 取-p~p的均值
            average_ST[0, model_id] = np.average(target_data.loc[10:30, col_name_ST])
            average_ST[1, model_id] = np.average(target_data[col_name_ST])
        
        average_ST_df = pd.DataFrame(average_ST, columns=model_type_list)
        average_ST_df.to_csv(root_save_floder + 'ST_average_' + direction_col + '_change_' + str(windows_len) + '_' + file_name + '.csv', header=True)

    return target_data





if __name__ == '__main__':

    root_save_floder = 'D:/HeatingOptimization/STCDRank/doc/excel/20220928/short_dingfu_low/window_size/'
    file_name = 'dingfu_high'
    windows_len = 6
    model_type_list = ['fitted_physical_model', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    calculate_trustworthiness(root_save_floder, model_type_list, file_name, windows_len)


'''
, SC_loss_weights, TC_loss_weights
TC_SC_loss_weight = TC_loss_weights * SC_loss_weights
                      'long_self_layer': TC_loss_weights,
                      'short_self_layer_other': SC_loss_weights,
                      'long_self_layer_other': TC_SC_loss_weight
'''