import numpy as np
from evaluation.metrics import save_score_predict_indoor_next
from common.config import baseline_trending_std_cols

#  evaluate、predict
def model_accuracy_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other):

    label_next_data, label_next_std_data, label_current_data = multi_time_test_label
    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_test_label
    # predict
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)
    #  save score
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_diff', indoor_temp_next_predict_diff, label_next_data) # 没有预测差值，用label_next_data占位。
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_next', indoor_temp_next_predict_truth, label_next_data)


    return indoor_temp_next_predict_diff, indoor_temp_next_predict_truth


def model_predict_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_label, multi_time_test_t_0_data, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other):


    label_next_data, label_next_std_data, label_current_data = multi_time_test_label
    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_test_label
    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_test_t_0_data

    # MDL
    continuous_baseline_day_data, continuous_baseline_week_data = MDL_baseline_data_test

    # other stations
    label_next_data_other, label_next_std_data_other, label_current_data_other = multi_time_test_t_0_label_other

    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_train_label
    continuous_model_data_other, continuous_baseline_data_other, wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other = multi_time_test_t_0_data_other

    continuous_baseline_day_data_other, continuous_baseline_week_data_other = MDL_baseline_data_test_other



    #  model.predict
    ## 模型变种
    if model_type == 'dnn_with_indoor_temp':
        print(model_type)
    ## baselines
    elif model_type in (['DNN_multi_time', 'LSTM']):
        predict_next_std = predict_model.predict([continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data])
        #print('predict_next_std\n', predict_next_std)
    elif model_type in (['MDL', 'MDL_DNN', 'MDL_LSTM']):
        #print(model_type)
        predict_next_std, predict_next_std_other = predict_model.predict([continuous_baseline_data, continuous_baseline_day_data, continuous_baseline_week_data, wind_data, weather_data, day_data, hour_data, havePeople_data, continuous_baseline_data_other, continuous_baseline_day_data_other, continuous_baseline_week_data_other, wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other])
    else:
        print('请检查模型设置')
    
    ## 恢复原始格式
    ## 反标准化
    if model_type in (['linear', 'DNN_multi_time', 'LSTM', 'MDL', 'MDL_DNN', 'MDL_LSTM']):
        predict_truth = label_next_ss.inverse_transform(predict_next_std)  #  反标准化
        #print('predict_truth\n', predict_truth)
        predict_truth = predict_truth.flatten()
        time_length = len(predict_truth)
        predict_diff = np.zeros((time_length), dtype=np.float)
    else:
        predict_diff = label_diff_ss.inverse_transform(predict_diff_std_short)  # 反标准化
        time_length = len(predict_diff)
        predict_truth = np.zeros((time_length), dtype=np.float)
        for i in range(0, time_length):
            #  左枝T时刻室温，加上预测室温差值，得到T+1时刻室温
            predict_truth[i] = label_current_data[i] + predict_diff[i]

    return predict_diff, predict_truth
