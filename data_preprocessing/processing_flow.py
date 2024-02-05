####################################################
# 数据预处理
## 第一步：读取数据，确定保存路径
## 第一步：特征筛选与构造
## 第二步：数据划分（train数据不能使用test时间的+n数据）
## 第三步：标签构造
## 第四步：去除异常值
## 第五步：数据标准化
####################################################
import pandas as pd
from data_preprocessing.data_IO import read_data
from data_preprocessing.drop_outliers import drop_nan_outliers, drop_outliers
from common.config import filter_cols, drop_outliers_clos, baseline_trending_numeric_cols, baseline_trending_std_cols, volatility_numeric_cols, volatility_std_cols, trending_numeric_cols, trending_std_cols
from data_preprocessing.label_feature import generate_next_label, generate_feature, get_current_short_long_data, get_difference_data
from data_preprocessing.split import split_data_no_fold
from data_preprocessing.standard import num_standard_label, get_trending_standard, get_volatility_standard, data_standard_transform, num_standard_feature_transform
from common.config import label_diff, label_diff_std


# 数据预处理
def data_preprocessing(preparation_floder, preprocessing_floder, file_name, fold_id, standard_type):

    ## 第一步：读取数据
    all_data = read_data(preparation_floder, file_name)
    #print(len(all_data))

    ## 第二步：去除异常值
    ### 如果供温、室温有一个为空，则删掉数据
    all_data = drop_nan_outliers(all_data)
    #print(len(all_data))
    ### 去掉真值3倍delta外数据
    all_data = drop_outliers(all_data, drop_outliers_clos)
    print(len(all_data))

    ## 第三步：缺失值填充
    all_data = all_data.fillna(method='ffill', axis = 0) # 每列用前一个值替换缺失值

    ## 第四步：特征筛选、真值标签（下一时刻室温）构造
    all_data = all_data[filter_cols]
    all_data = generate_next_label(all_data)
    ###  先对室温真值做标准化，用于baselines
    label_next_ss, all_data = num_standard_label(all_data, "indoor_temp_next", "indoor_temp_next_std", standard_type)
    #print(all_data)
    #all_data.to_csv(preprocessing_floder + 'all_data/' + file_name + '.csv', header=True, index=False)

    ## 第五步：数据划分（train数据不能使用test时间的+n数据）
    ### 6:2:2划分，先取20%作为测试集
    #train_data, test_data = split_data(all_data, 0.2)
    #train_data, test_data = get_train_test_fold(all_data, fold_id)
    train_data, test_data = split_data_no_fold(all_data, 0.2)

    ## 第六步：针对我们的模型，构造差值标签、趋势差值特征、波动差值特征。
    ### 需按长短期分类。train需长短期数据，test只需处理短期特征。
    train_current_data, train_short_data, train_long_data = get_current_short_long_data(train_data)
    train_trending_data, train_short_volatility, train_long_volatility = get_difference_data(train_current_data, train_short_data, train_long_data)
    test_current_ori, test_short_ori, test_long_ori = get_current_short_long_data(test_data)
    test_trending_data, test_short_volatility, test_long_volatility = get_difference_data(test_current_ori, test_short_ori, test_long_ori)
    
    ## 第七步：数据标准化
    trending_feature_ss, train_trending_data, test_trending_data = get_trending_standard(train_trending_data, test_trending_data, baseline_trending_numeric_cols, baseline_trending_std_cols, standard_type)
    label_diff_ss, volatility_feature_ss, train_short_volatility, train_long_volatility, test_short_volatility, test_long_volatility = get_volatility_standard(train_short_volatility, train_long_volatility, test_short_volatility, test_long_volatility, volatility_numeric_cols, volatility_std_cols, standard_type)
    #print(train_trending_data)
    #print(train_short_volatility)
    #print(all_data)

    return label_next_ss, label_diff_ss, trending_feature_ss, volatility_feature_ss, train_trending_data, train_short_volatility, train_long_volatility, test_trending_data, test_short_volatility, test_long_volatility, test_current_ori, test_short_ori, test_long_ori

def get_other_stations_data(preprocessing_floder, file_list, file_name, fold_id, label_diff_ss, trending_feature_ss, volatility_feature_ss, train_data_sample_len, test_data_sample_len):

    other_stations_train_current = pd.DataFrame()
    other_stations_train_short = pd.DataFrame()
    other_stations_train_long = pd.DataFrame()
    other_stations_test_current = pd.DataFrame()
    other_stations_test_short = pd.DataFrame()
    other_stations_test_long = pd.DataFrame()
    # 读取和划分数据
    for file_id in range(0, len(file_list)):
        current_file_name = file_list[file_id]
        #print(current_file_name)
        #print(file_name)
        if current_file_name != file_name:
            current_file_data = pd.read_csv(preprocessing_floder + 'all_data/' + current_file_name + '.csv', index_col=False)
            #train_data, test_data = get_train_test_fold(current_file_data, fold_id)
            train_data, test_data = split_data_no_fold(current_file_data, 0.2)
            train_current_data, train_short_data, train_long_data = get_current_short_long_data(train_data)
            test_current_data, test_short_data, test_long_data = get_current_short_long_data(test_data)
            train_trending_data, train_short_volatility, train_long_volatility = get_difference_data(train_current_data, train_short_data, train_long_data)
            test_trending_data, test_short_volatility, test_long_volatility = get_difference_data(test_current_data, test_short_data, test_long_data)
            other_stations_train_current = other_stations_train_current.append(train_trending_data)
            other_stations_train_short = other_stations_train_short.append(train_short_volatility)
            other_stations_train_long = other_stations_train_long.append(train_long_volatility)
            other_stations_test_current = other_stations_test_current.append(test_trending_data)
            other_stations_test_short = other_stations_test_short.append(test_short_volatility)
            other_stations_test_long = other_stations_test_long.append(test_long_volatility)
    # 抽样
    other_stations_train_current = other_stations_train_current.sample(n=train_data_sample_len).reset_index(drop=True)
    other_stations_train_short = other_stations_train_short.sample(n=train_data_sample_len).reset_index(drop=True)
    other_stations_train_long = other_stations_train_long.sample(n=train_data_sample_len).reset_index(drop=True)
    other_stations_test_current = other_stations_test_current.sample(n=test_data_sample_len).reset_index(drop=True)
    other_stations_test_short = other_stations_test_short.sample(n=test_data_sample_len).reset_index(drop=True)
    other_stations_test_long = other_stations_test_long.sample(n=test_data_sample_len).reset_index(drop=True)
    
    # 标准化
    other_stations_train_current = num_standard_feature_transform(other_stations_train_current, trending_feature_ss, baseline_trending_numeric_cols, baseline_trending_std_cols)
    other_stations_train_short = data_standard_transform(other_stations_train_short, label_diff_ss, volatility_feature_ss, label_diff, label_diff_std, volatility_numeric_cols, volatility_std_cols)
    other_stations_train_long = data_standard_transform(other_stations_train_long, label_diff_ss, volatility_feature_ss, label_diff, label_diff_std, volatility_numeric_cols, volatility_std_cols)
    other_stations_test_current = num_standard_feature_transform(other_stations_test_current, trending_feature_ss, baseline_trending_numeric_cols, baseline_trending_std_cols)
    other_stations_test_short = data_standard_transform(other_stations_test_short, label_diff_ss, volatility_feature_ss, label_diff, label_diff_std, volatility_numeric_cols, volatility_std_cols)
    other_stations_test_long = data_standard_transform(other_stations_test_long, label_diff_ss, volatility_feature_ss, label_diff, label_diff_std, volatility_numeric_cols, volatility_std_cols)


    return other_stations_train_current, other_stations_train_short, other_stations_train_long, other_stations_test_current, other_stations_test_short, other_stations_test_long


if __name__ == '__main__':

    preparation_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preparation/original_indoor_temp/'
    preprocessing_floder = 'D:/HeatingOptimization/MultipleMoments/result/data_preprocessing/original_indoor_temp/'
    file_name = 'dingfu_high'
    standard_type = 'standard'

    all_data = data_preprocessing(preparation_floder, preprocessing_floder, file_name, standard_type)

