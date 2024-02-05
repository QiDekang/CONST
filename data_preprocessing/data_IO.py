import os
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd


def read_data(root_read_floder, file_name):

    #  data
    read_file_path = root_read_floder + file_name + '_data.csv'
    data = pd.read_csv(read_file_path, index_col=False)
    #print('data.columns:', data.columns.values.tolist())

    return data

def get_save_folder(root_save_floder, model_type, file_name, repeat_id, fold_id):

    folder_path = root_save_floder + model_type
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    #save_folder = root_save_floder + model_type + '/' + file_name + '_' + \
    #    'repeat_' + str(repeat_id) + '_'
    save_folder = root_save_floder + model_type + '/' + file_name + '_' + 'repeat_' + str(repeat_id) + '_fold_' + str(fold_id) + '_'

    return save_folder

def save_mid_data(root_save_floder, file_name, repeat_id):

    # 构造长期数据有随机数，重复三遍；
    # 模型训练有随机数，重复三遍；
    # 共9遍

    return root_save_floder
