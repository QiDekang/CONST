import numpy as np
from evaluation.metrics import save_score_predict_indoor_next
from common.config import baseline_trending_std_cols

#  evaluate、predict

def model_accuracy_F_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test):

    #label_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label_test
    label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0 = multi_time_test_t_0_label

    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_F_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test)
    
    #  save score
    #save_score_predict_indoor_next(
    #    save_folder, 'indoor_temp_diff', indoor_temp_next_predict_diff, label_diff)
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_next', indoor_temp_next_predict_truth, label_next_data_t_0)

    return indoor_temp_next_predict_diff, indoor_temp_next_predict_truth


def model_predict_F_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test):


    # 解压数据
    label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0 = multi_time_test_t_0_label
    continuous_model_data_t_0, continuous_baseline_data_t_0, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0 = multi_time_test_t_0_data
    label_next_data_t_n, label_next_std_data_t_n, label_current_data_t_n = multi_time_test_t_n_label
    continuous_model_data_t_n, continuous_baseline_data_t_n, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = multi_time_test_t_n_data

    #  其他站点数据
    multi_time_train_t_0_label_enhancement_other, multi_time_train_t_0_data_enhancement_other, multi_time_train_t_n_label_other, multi_time_train_t_n_data_other, temporal_discount_rate_other = multi_time_0_1_n_data_other_test
    label_next_data_t_0_other, label_next_std_data_t_0_other, label_current_data_t_0_other = multi_time_train_t_0_label_enhancement_other
    continuous_model_data_t_0_other, continuous_baseline_data_t_0_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other = multi_time_train_t_0_data_enhancement_other
    label_next_data_t_n_other, label_next_std_data_t_n_other, label_current_data_t_n_other = multi_time_train_t_n_label_other
    continuous_model_data_t_n_other, continuous_baseline_data_t_n_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other = multi_time_train_t_n_data_other

    '''
    label_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label_test


    continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data_std_test
    continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = long_data_std_test


    # other 
    short_label_test_other, short_data_std_test_other, long_data_std_test_other, temporal_discount_rate_test_other = accuracy_data_other
    continuous_data_short_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other = short_data_std_test_other
    continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other = long_data_std_test_other
    '''

    #  model.predict
    ## 模型变种
    if model_type == 'STCD_LSTM_F_TC_loss_SC':
        predict_next_std, predict_next_t_n_std_discount, predict_next_std_other, predict_next_t_n_std_discount_other = predict_model.predict([temporal_discount_rate_test, continuous_baseline_data_t_0, continuous_baseline_data_t_n, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n, temporal_discount_rate_other, continuous_baseline_data_t_0_other, continuous_baseline_data_t_n_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other])
    
    
    ## 恢复原始格式
    ## 反标准化
    #if model_type == 'dnn_with_indoor_temp' or model_type == 'dnn_without_indoor_temp' or model_type == 'linear_without_indoor_temp':
    if model_type in (['STCF_MFF', 'linear', 'DNN', 'DNN_with_embedding', 'LSTM', 'STCD_LSTM_F_TC_loss_SC']):
        predict_truth = label_next_ss.inverse_transform(predict_next_std)  #  反标准化
        predict_truth = predict_truth.flatten()
        time_length = np.size(predict_truth, 0)
        # 占位
        predict_diff = np.zeros((time_length), dtype=np.float)
    else:
        predict_diff = label_diff_ss.inverse_transform(predict_diff_std_short)  # 反标准化
        time_length = np.size(predict_diff, 0)
        predict_truth = np.zeros((time_length), dtype=np.float)
        for i in range(0, time_length):
            #  左枝T时刻室温，加上预测室温差值，得到T+1时刻室温
            predict_truth[i] = label_current_data_t_0[i] + predict_diff[i]

    return predict_diff, predict_truth
