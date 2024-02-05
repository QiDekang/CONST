file_list_all = ["qintaoyuan_low", "qintaoyuan_high", "dingfu_high", "dingfu_low", "hangda",
                 "zhangjiawo", "zuanshishan_high", "zuanshishan_low", "zuanshishan_mid", "haihe_high"]
file_list_five = ["qintaoyuan_low", "qintaoyuan_high",
                  "dingfu_high", "dingfu_low", "haihe_high"]
#file_list_four = ["qintaoyuan_low", "qintaoyuan_high",
#                  "dingfu_high", "dingfu_low"]
file_list_four = ["dingfu_high", "dingfu_low", "qintaoyuan_high", "qintaoyuan_low"]

#file_list_three = ["dingfu_high", "dingfu_low", "qintaoyuan_low"]

#file_list_three_q_l = ["dingfu_high", "dingfu_low", "qintaoyuan_high"]
file_list = ["dingfu_high", "dingfu_low", "qintaoyuan_high"]

#filter_cols = ['time', 'indoor_temp', 'second_heat_temp', 'second_heat_pressure', 'illumination', 'outdoor_temp', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed', 'wind_direction', 'weather', 'day', 'hour', 'havePeople']
filter_cols = ['time', 'indoor_temp', 'second_heat_temp', 'second_heat_pressure', 'second_return_temp', 'second_return_pressure', 'illumination', 'outdoor_temp', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed', 'wind_direction', 'weather', 'day', 'hour', 'havePeople']

# 根据真值，去除异常值
drop_outliers_clos = ['indoor_temp', 'second_heat_temp', 'illumination', 'outdoor_temp', 'wind_speed']

# 随机的单位时间应在72小时以内
#max_unit_time = 72
max_unit_time = 2000

# 趋势数据
trending_cols = ['time', 'indoor_temp_next', 'heat_indoor_temp', 'indoor_outdoor_temp', 'second_heat_pressure', 'illumination', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed', 'wind_direction', 'weather', 'day', 'hour', 'havePeople']

# 需标准化的列
#trending_numeric_cols = ['heat_indoor_temp', 'indoor_outdoor_temp', 'second_heat_pressure', 'illumination', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed']
#trending_std_cols = ['heat_indoor_temp_std', 'indoor_outdoor_temp_std', 'second_heat_pressure_std', 'illumination_std', 'outdoor_pressure_std', 'outdoor_humidity_std', 'wind_speed_std']
# baseline需要用室温、供温、外温，baseline和和trending所有的列
#baseline_trending_numeric_cols = ['indoor_temp', 'second_heat_temp', 'outdoor_temp', 'heat_indoor_temp', 'indoor_outdoor_temp', 'second_heat_pressure', 'illumination', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed']
#baseline_trending_std_cols = ['indoor_temp_std', 'second_heat_temp_std', 'outdoor_temp_std', 'heat_indoor_temp_std', 'indoor_outdoor_temp_std', 'second_heat_pressure_std', 'illumination_std', 'outdoor_pressure_std', 'outdoor_humidity_std', 'wind_speed_std']
#volatility_numeric_cols = ['heat_indoor_temp_diff', 'indoor_outdoor_temp_diff', 'second_heat_pressure_diff', 'illumination_diff', 'outdoor_pressure_diff', 'outdoor_humidity_diff', 'wind_speed_diff']
#volatility_std_cols = ['heat_indoor_temp_diff_std', 'indoor_outdoor_temp_diff_std', 'second_heat_pressure_diff_std', 'illumination_diff_std', 'outdoor_pressure_diff_std', 'outdoor_humidity_diff_std', 'wind_speed_diff_std']

#baseline_trending_numeric_cols = ['indoor_temp', 'second_heat_temp', 'outdoor_temp', 'second_heat_pressure', 'illumination', 'outdoor_pressure', 'outdoor_humidity', 'wind_speed']
#baseline_trending_std_cols = ['indoor_temp_std', 'second_heat_temp_std', 'outdoor_temp_std', 'second_heat_pressure_std', 'illumination_std', 'outdoor_pressure_std', 'outdoor_humidity_std', 'wind_speed_std']
# 波动网络
# 全部特征
#volatility_numeric_cols = ['second_heat_temp_diff', 'outdoor_temp_diff', 'second_heat_pressure_diff', 'illumination_diff', 'outdoor_pressure_diff', 'outdoor_humidity_diff', 'wind_speed_diff']
#volatility_std_cols = ['second_heat_temp_diff_std', 'outdoor_temp_diff_std', 'second_heat_pressure_diff_std', 'illumination_diff_std', 'outdoor_pressure_diff_std', 'outdoor_humidity_diff_std', 'wind_speed_diff_std']
# 部分特征
volatility_numeric_cols = ['second_heat_temp_diff', 'outdoor_temp_diff', 'illumination_diff', 'wind_speed_diff']
volatility_std_cols = ['second_heat_temp_diff_std', 'outdoor_temp_diff_std', 'illumination_diff_std', 'wind_speed_diff_std']
# 生成特征
#volatility_numeric_cols = ['heat_indoor_temp_diff', 'indoor_outdoor_temp_diff', 'illumination_diff', 'wind_speed_diff']
#volatility_std_cols = ['heat_indoor_temp_diff_std', 'indoor_outdoor_temp_diff_std', 'illumination_diff_std', 'wind_speed_diff_std']
# 趋势特征
trending_numeric_cols = ['second_heat_temp', 'outdoor_temp', 'illumination', 'wind_speed']
trending_std_cols = ['second_heat_temp_std', 'outdoor_temp_std', 'illumination_std', 'wind_speed_std']

# baselines所需特征
baseline_trending_numeric_cols = ['indoor_temp', 'second_heat_temp', 'outdoor_temp', 'illumination', 'wind_speed']
baseline_trending_std_cols = ['indoor_temp_std', 'second_heat_temp_std', 'outdoor_temp_std', 'illumination_std', 'wind_speed_std']

#  dropout_rate
dropout_rate = 0.1

#  日志显示级别
verbose_level = 0
#verbose_level = 1
#verbose_level = 2

label_next_std = ['indoor_temp_next_std']
label_diff = 'indoor_temp_diff'
label_diff_std = 'indoor_temp_diff_std'

# LSTM 时间窗口
window_size = 24
# MMF的时间窗口，使用最近3个小时的数据，k+1=3
seq_len = 2



#############################################
baseline_numeric_cols = ['indoor_temp', 'second_heat_temp', 'outdoor_temp', 'illumination', 'wind_speed']
baseline_std_cols = ['indoor_temp_std', 'second_heat_temp_std', 'outdoor_temp_std', 'illumination_std', 'wind_speed_std']

model_numeric_cols = ['second_heat_temp', 'outdoor_temp', 'illumination', 'wind_speed']
model_std_cols = ['second_heat_temp_std', 'outdoor_temp_std', 'illumination_std', 'wind_speed_std']



label_next = 'indoor_temp_next'
label_next_std = 'indoor_temp_next_std'
label_diff = 'indoor_temp_diff'
label_diff_std = 'indoor_temp_diff_std'
label_current = 'indoor_temp'

#discrete_feature_num = 5  #离散特征数量
discrete_feature_num = 4  #离散特征数量
embed_size = 2  #离散特征嵌入后输出的维度。
#embed_size = 3  #离散特征嵌入后输出的维度。


# pairwise数据n*n个，太多，时间太长，抽样部分数据1%约相当于15*n
#sampling_ratio = 0.001
sampling_ratio = 0.01


#close_effect_rate = 0.97
#periodic_effect_rate = 0.97

#close_effect_rate = 0.98
#periodic_effect_rate = 0.98

#close_effect_rate = 0.95
#periodic_effect_rate = 0.95

close_effect_rate = 1
periodic_effect_rate = 1

random_seed = 2022

l2_rate = 0.01

learning_rate = 0.0001

#SC_loss_weights = 0.5
SC_loss_weights = 1


PDP_grid_resolution = 100
