import numpy as np
from evaluation.metrics import save_score_predict_indoor_next
from common.config import baseline_trending_std_cols
from baselines.fitted_physical_model import get_fitted_physical_model

#  evaluate、predict
def model_accuracy_physical(save_folder, model_type, label_next_ss, test_data, file_name, is_fitted):

    label_next_data = test_data['indoor_temp_next']
    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_test_label
    # predict
    indoor_temp_next_predict_diff, indoor_temp_next_predict_truth = model_predict_physical(model_type, label_next_ss, test_data, file_name, is_fitted)
    #  save score
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_diff', indoor_temp_next_predict_diff, label_next_data) # 没有预测差值，用label_next_data占位。
    save_score_predict_indoor_next(
        save_folder, 'indoor_temp_next', indoor_temp_next_predict_truth, label_next_data)


    return indoor_temp_next_predict_diff, indoor_temp_next_predict_truth


def model_predict_physical(model_type, label_next_ss, test_data, file_name, is_fitted):

    #  model.predict
    if model_type == 'Fitted_physical_model':
        #print(model_type)
        predict_next = get_fitted_physical_model(test_data, file_name, is_fitted)

    predict_truth = predict_next
    predict_diff = np.zeros(len(predict_next), dtype=np.float)
    '''
    ## 恢复原始格式
    ## 反标准化
    if model_type in (['linear', 'DNN_multi_time', 'LSTM', 'ResNet']):
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
    '''

    return predict_diff, predict_truth
