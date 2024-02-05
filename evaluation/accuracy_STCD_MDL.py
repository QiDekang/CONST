import numpy as np
from evaluation.metrics import save_score_predict_indoor_next
from common.config import baseline_trending_std_cols

#  evaluate、predict

def model_accuracy_STCD_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other):

    label_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label_test
    
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_STCD_MDL(model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other)
    
    #  save score
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_diff', indoor_temp_next_predict_diff, label_diff)
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_next', indoor_temp_next_predict_truth, label_next_data_t_0)

    return indoor_temp_next_predict_diff, indoor_temp_next_predict_truth


def model_predict_STCD_MDL(model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other):

    label_diff, label_next_data_t_0, label_next_std_data_t_0, label_current_data_t_0, label_next_data_t_1, label_next_std_data_t_1, label_current_data_t_1 = short_label_test


    continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data_std_test
    continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = long_data_std_test


    # other 
    short_label_test_other, short_data_std_test_other, long_data_std_test_other, temporal_discount_rate_test_other = accuracy_data_other
    continuous_data_short_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other = short_data_std_test_other
    continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other = long_data_std_test_other

    # MDL数据
    continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std = MDL_model_data_need_test
    continuous_model_day_short_diff_std_other, continuous_model_day_long_diff_std_other, continuous_model_week_short_diff_std_other, continuous_model_week_long_diff_std_other = MDL_model_data_need_test_other


    #  model.predict
    ## 模型变种
    if model_type == 'STCD_DNN_DF_TC':
        predict_diff_std_short, label_t_n_t_0_diff_std_discount = predict_model.predict([continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n])
    elif model_type == 'STCD_DNN_DF_TC_loss':
        predict_diff_std_short, label_t_n_t_0_diff_std_discount = predict_model.predict([temporal_discount_rate_test, continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n])
    elif model_type in ('STCD_DNN_DF_TC_loss_SC', 'STCD_LSTM_DF_TC_loss_SC'):
        predict_diff_std_short, label_t_n_t_0_diff_std_discount, predict_diff_std_short_other, label_t_n_t_0_diff_std_discount_other = predict_model.predict([temporal_discount_rate_test, continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n, temporal_discount_rate_test_other, continuous_data_short_diff_all_std_other, continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other])
    elif model_type in ('STCD_MDL_all'):
        predict_diff_std_short, label_t_n_t_0_diff_std_discount, predict_diff_std_short_other, label_t_n_t_0_diff_std_discount_other = predict_model.predict([temporal_discount_rate_test, continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n, temporal_discount_rate_test_other, continuous_data_short_diff_all_std_other, continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other, continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std, continuous_model_day_short_diff_std_other, continuous_model_day_long_diff_std_other, continuous_model_week_short_diff_std_other, continuous_model_week_long_diff_std_other])


    ## 恢复原始格式
    ## 反标准化
    #if model_type == 'dnn_with_indoor_temp' or model_type == 'dnn_without_indoor_temp' or model_type == 'linear_without_indoor_temp':
    if model_type in (['STCF_MFF', 'linear', 'DNN', 'DNN_with_embedding', 'LSTM']):
        predict_truth = label_next_ss.inverse_transform(predict_next_std)  #  反标准化
        predict_truth = predict_truth.flatten()
        time_length = np.size(predict_truth, 0)
        predict_diff = np.zeros((time_length), dtype=np.float)
    else:
        predict_diff = label_diff_ss.inverse_transform(predict_diff_std_short)  # 反标准化
        time_length = np.size(predict_diff, 0)
        predict_truth = np.zeros((time_length), dtype=np.float)
        for i in range(0, time_length):
            #  左枝T时刻室温，加上预测室温差值，得到T+1时刻室温
            predict_truth[i] = label_current_data_t_0[i] + predict_diff[i]

    return predict_diff, predict_truth
