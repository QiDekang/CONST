########################################################
# 数据预处理

# 模型训练
## baselines
## 模型及消融实验

# 效果评估
## 预测准确度（应计算1、2、3、4个小时的准确度）
## 预测可信度（方向与幅度）
########################################################
from data_preprocessing.processing_flow import data_preprocessing, get_other_stations_data
from model.variant_model.long_short_volatility import model_train_long_short_volatility
from common.config import volatility_numeric_cols, volatility_std_cols, label_diff_std, trending_std_cols, label_next_std, trending_std_cols, baseline_trending_std_cols
from data_preprocessing.data_IO import get_save_folder
#from evaluation.accuracy import model_accuracy
#from evaluation.direction import is_direction_right_all
from model.variant_model.spatial_consistent import model_train_spatial_consistent
from model.variant_model.STCF_MFF import model_train_STCF_MFF
from model.variant_model.STCF_MFF_D import model_train_STCF_MFF_D
from model.variant_model.STCF_MFF_D_TC import model_train_STCF_MFF_D_TC
from model.variant_model.STCF_MFF_D_TC_MMF import model_train_STCF_MFF_D_TC_MMF
from data_preprocessing.MMF_data import get_MMF_data
#from evaluation.accuracy_MMF import model_accuracy_MMF
#from evaluation.direction_MMF import is_direction_right_all_MMF
from model.variant_model.STCF_all import model_train_STCF_all
#from evaluation.accuracy_STCF_all import model_accuracy_STCF_all
#from evaluation.direction_STCF_all import is_direction_right_all_STCF_all


from baselines.linear import model_train_linear
from baselines.LSTM import model_train_LSTM
from data_preprocessing.LSTM_data import get_lstm_data
#from evaluation.accuracy_LSTM import model_accuracy_LSTM
#from evaluation.direction_LSTM import is_direction_right_all_LSTM
from common.config import window_size, seq_len

from data_preprocessing.flow import get_data, get_all_other_stations_data
from data_preprocessing.flow import get_multi_time_data, get_long_short_data, get_all_diff_data, get_diff_std, get_diff_std_test
import numpy as np
from common.config import model_std_cols, baseline_std_cols
from evaluation.accuracy_DF_TC import model_accuracy_DF_TC
from evaluation.trustworthiness_DF_TC import trustworthiness_all_DF_TC
from STCD_models.modules.temporal_discount import get_temporal_discount_shuffle_data
from evaluation.accuracy_DF_TC_loss import model_accuracy_DF_TC_loss
from evaluation.trustworthiness_DF_TC_loss import trustworthiness_all_DF_TC_loss
from STCD_models.variant_models.STCD_DNN_DF_TC_loss_SC import model_train_STCD_DNN_DF_TC_loss_SC
from evaluation.accuracy_DF_TC_loss_SC import model_accuracy_DF_TC_loss_SC
from evaluation.trustworthiness_DF_TC_loss_SC import trustworthiness_all_DF_TC_loss_SC
from STCD_models.variant_models.STCD_LSTM_DF_TC_loss_SC import model_train_STCD_LSTM_DF_TC_loss_SC
from baselines.DNN_multi_time import model_train_DNN_multi_time
from evaluation.accuracy_baselines import model_accuracy_baselines
from evaluation.trustworthiness_baselines import model_trustworthiness_all_baselines
from data_preprocessing.flow import get_MDL_data
from baselines.MDL_DNN import model_train_MDL_DNN
from baselines.MDL_LSTM import model_train_MDL_LSTM
from evaluation.accuracy_MDL import model_accuracy_MDL
from evaluation.trustworthiness_MDL import model_trustworthiness_all_MDL
from STCD_models.variant_models.STCD_MDL_DF_TC_loss_SC import model_train_STCD_MDL_DF_TC_loss_SC
from evaluation.accuracy_STCD_MDL import model_accuracy_STCD_MDL
from evaluation.trustworthiness_STCD_MDL import trustworthiness_all_STCD_MDL
from baselines.linear import model_train_linear
from STCD_models.variant_models.STCD_LSTM_DF_TC_loss import model_train_STCD_LSTM_DF_TC_loss
from STCD_models.variant_models.STCD_LSTM_DF import model_train_STCD_LSTM_DF
from STCD_models.variant_models.STCD_ResNet_all import model_train_STCD_ResNet_all
from data_preprocessing.flow import get_long_short_data_enhancement
from data_preprocessing.flow import get_MDL_data_enhancement
from baselines.ResNet import model_train_ResNet
from STCD_models.variant_models.STCD_LSTM_DF_SC import model_train_STCD_LSTM_DF_SC
from data_preprocessing.flow import get_multi_time_data_wo_DF
from evaluation.accuracy_linear import model_accuracy_linear
from evaluation.trustworthiness_linear import model_trustworthiness_all_linear
from baselines.fitted_physical_model import get_fitted_physical_model
from evaluation.accuracy_physical import model_accuracy_physical
from evaluation.trustworthiness_physical import model_trustworthiness_all_physical
from baselines.fitted_physical_model_prepare import fitted_physical_model_prepare
from STCD_models.variant_models.STCD_LSTM_DF_TC_SC import model_train_STCD_LSTM_DF_TC_SC
from STCD_models.variant_models.STCD_LSTM_continuous import model_train_STCD_LSTM_continuous
from STCD_models.variant_models.STCD_LSTM_F_TC_loss_SC import model_train_STCD_LSTM_F_TC_loss_SC
from evaluation.accuracy_F_TC_loss_SC import model_accuracy_F_TC_loss_SC
from evaluation.trustworthiness_F_TC_loss_SC import trustworthiness_all_F_TC_loss_SC

from evaluation.trustworthiness_DF_TC_loss_SC_PDP import trustworthiness_all_DF_TC_loss_SC_PDP
from evaluation.trustworthiness_F_TC_loss_SC_PDP import trustworthiness_all_F_TC_loss_SC_PDP
from evaluation.trustworthiness_baselines_PDP import model_trustworthiness_all_baselines_PDP
from evaluation.trustworthiness_MDL_PDP import model_trustworthiness_all_MDL_PDP
from evaluation.trustworthiness_STCD_MDL_PDP import trustworthiness_all_STCD_MDL_PDP
from evaluation.trustworthiness_linear_PDP import model_trustworthiness_all_linear_PDP
from evaluation.trustworthiness_physical_PDP import model_trustworthiness_all_physical_PDP




def process(preparation_floder, preprocessing_floder, root_save_floder, file_list, file_name, repeat_id, fold_id, model_type_list, standard_type, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, trend_effect_rate, SC_loss_weights, TC_loss_weights):


    #print('file_name: ', file_name)
    #print('repeat_id:', repeat_id)
    #print('fold_id:', fold_id)
    #print('enhancement_times', enhancement_times)

    # 数据准备

    label_next_ss, feature_ss, train_data, test_data = get_data(preparation_floder, preprocessing_floder, file_name, fold_id, standard_type, long_predict_len)
    fitted_physical_data = train_data[['second_heat_temp', 'second_return_temp', 'outdoor_temp', 'indoor_temp_next']]
    #fitted_physical_data.to_csv(preprocessing_floder + 'fitted_physical_data/' + file_name + '.csv', header=True, index=False)
    
    #print('train_data', train_data)
    train_time_len = np.size(train_data, 0)
    test_time_len = np.size(test_data, 0)

    multi_time_train_t_0_label, multi_time_train_t_0_data = get_multi_time_data(train_data, windows_len)
    multi_time_test_t_0_label, multi_time_test_t_0_data = get_multi_time_data(test_data, windows_len)

    
    #print('multi_time_train_t_0_data', multi_time_train_t_0_data)
    # t-1, t-n
    ## 将训练数据增强，50倍、100倍、n倍等。测试数据不用增强。
    multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, long_index_array = get_long_short_data(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len)

    #print('enhancement_times', enhancement_times)
    multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data, current_index_array, short_index_array, long_index_array = get_long_short_data_enhancement(multi_time_train_t_0_label, multi_time_train_t_0_data, train_time_len, enhancement_times)
    multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data, long_index_array_test = get_long_short_data(multi_time_test_t_0_label, multi_time_test_t_0_data, test_time_len)

    # 差分
    
    #short_label, short_data, long_label, long_data = get_all_diff_data(multi_time_train_t_0_label, multi_time_train_t_0_data, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data)
    short_label, short_data, long_label, long_data = get_all_diff_data(multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_1_label, multi_time_train_t_1_data, multi_time_train_t_n_label, multi_time_train_t_n_data)
    short_label_test, short_data_test, long_label_test, long_data_test = get_all_diff_data(multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data)

    # 标准化
    label_short_diff_std, label_long_diff_std, label_diff_ss, short_data_std, long_data_std, feature_diff_ss = get_diff_std(short_label, short_data, long_label, long_data, standard_type, windows_len)
    # test只标准化特征即可，预测出标签
    short_data_std_test, long_data_std_test = get_diff_std_test(short_data_test, long_data_test, windows_len, feature_diff_ss)


    ### 构造时间差数据和时间贴现数据
    #### 需将预测和真实的差值标签乘以temporal_discount_rate
    temporal_discount_rate = get_temporal_discount_shuffle_data(train_data, current_index_array, long_index_array, close_effect_rate, periodic_effect_rate, trend_effect_rate)
    current_index_array_test = np.array(range(0, len(test_data)))
    #print('current_index_array_test',current_index_array_test)
    temporal_discount_rate_test = get_temporal_discount_shuffle_data(test_data, current_index_array_test, long_index_array_test, close_effect_rate, periodic_effect_rate, trend_effect_rate)


    # MDL相关数据
    #MDL_baseline_data, MDL_model_data_need = get_MDL_data(train_data, train_time_len, long_index_array, windows_len, feature_diff_ss)
    MDL_baseline_data, MDL_model_data_need = get_MDL_data_enhancement(train_data, train_time_len, current_index_array, short_index_array, long_index_array, windows_len, feature_diff_ss)
    MDL_baseline_data_test, MDL_model_data_need_test = get_MDL_data(test_data, test_time_len, long_index_array_test, windows_len, feature_diff_ss)
    #print('MDL_baseline_data\n', MDL_baseline_data)


    # 获取其他站点数据，合并成一份数据
    if len(file_list) < 2:
        print('仅有一个数据集')
    else:
        model_train_data_other, accuracy_data_other, multi_time_0_1_n_data_other, train_data_other, test_data_other, multi_time_0_1_n_data_other_test = get_all_other_stations_data(preparation_floder, preprocessing_floder, file_list, file_name, fold_id, label_next_ss, label_diff_ss, feature_ss, feature_diff_ss, train_time_len, test_time_len, windows_len, enhancement_times, long_predict_len, close_effect_rate, periodic_effect_rate, trend_effect_rate)
        #print('train_data_other', train_data_other)
        
        # MDL相关数据
        #MDL_baseline_data_other, MDL_model_data_need_other = get_MDL_data(train_data_other, train_time_len, long_index_array, windows_len, feature_diff_ss)
        MDL_baseline_data_other, MDL_model_data_need_other = get_MDL_data_enhancement(train_data_other, train_time_len, current_index_array, short_index_array, long_index_array, windows_len, feature_diff_ss)
        MDL_baseline_data_test_other, MDL_model_data_need_test_other = get_MDL_data(test_data_other, test_time_len, long_index_array_test, windows_len, feature_diff_ss)
        #print('MDL_baseline_data_other\n', MDL_baseline_data_other)
        # baselines使用的数据不用增强
        multi_time_train_t_0_label_other, multi_time_train_t_0_data_other = get_multi_time_data(train_data_other, windows_len)
        multi_time_test_t_0_label_other, multi_time_test_t_0_data_other = get_multi_time_data(test_data_other, windows_len)


    

    # 模型
    for model_id in range(0, len(model_type_list)):  # 不同方法
        model_type = model_type_list[model_id]
        print('model_type: ', model_type)
        ## 确定保存路径
        save_folder = get_save_folder(root_save_floder, model_type, file_name, repeat_id, fold_id)


        if model_type in ('STCD_DNN_DF_TC_loss_SC', 'STCD_LSTM_DF_TC_loss_SC', 'STCD_LSTM_DF_TC_loss', 'STCD_LSTM_DF_TC', 'STCD_LSTM_DF', 'STCD_ResNet_all', 'STCD_LSTM_DF_SC', 'STCD_LSTM_DF_TC_SC', 'STCD_LSTM_continuous'):
            if model_type == 'STCD_DNN_DF_TC_loss_SC':
                predict_model = model_train_STCD_DNN_DF_TC_loss_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_DF_TC_loss_SC':
                # 仅在STCD_LSTM_DF_TC_loss_SC中调节SC_loss_weights, TC_loss_weights参数，调整好后，其他模型直接在config中读取即可。
                predict_model = model_train_STCD_LSTM_DF_TC_loss_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_DF_TC_SC':
                predict_model = model_train_STCD_LSTM_DF_TC_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_continuous':
                predict_model = model_train_STCD_LSTM_continuous(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_DF_TC_loss':
                predict_model = model_train_STCD_LSTM_DF_TC_loss(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_DF_SC':
                predict_model = model_train_STCD_LSTM_DF_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_LSTM_DF':
                predict_model = model_train_STCD_LSTM_DF(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
            elif model_type == 'STCD_ResNet_all':
                predict_model = model_train_STCD_ResNet_all(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, SC_loss_weights, TC_loss_weights)
                                             
            
            # 精度
            model_accuracy_DF_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test, accuracy_data_other)
            # 可信度
            multi_time_0_1_n_data = [multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data]
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                trustworthiness_all_DF_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other)
                trustworthiness_all_DF_TC_loss_SC_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other)
                

        # withou Difference，long相当于将short重复一遍。
        elif model_type == 'STCD_LSTM_F_TC_loss_SC':
            predict_model = model_train_STCD_LSTM_F_TC_loss_SC(save_folder, multi_time_train_t_0_label_enhancement, multi_time_train_t_0_data_enhancement, multi_time_train_t_n_label, multi_time_train_t_n_data, baseline_std_cols, windows_len, temporal_discount_rate, model_train_data_other, multi_time_0_1_n_data_other, SC_loss_weights, TC_loss_weights)
            
            model_accuracy_F_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test)
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                trustworthiness_all_F_TC_loss_SC(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test, feature_ss)
                trustworthiness_all_F_TC_loss_SC_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_n_label, multi_time_test_t_n_data, temporal_discount_rate_test, multi_time_0_1_n_data_other_test, feature_ss)


        ## baseline
        elif model_type in ('DNN_multi_time', 'LSTM', 'ResNet'):
            if model_type == 'DNN_multi_time':
                #print('DNN_multi_time')
                predict_model = model_train_DNN_multi_time(save_folder, multi_time_train_t_0_label, multi_time_train_t_0_data, baseline_std_cols, windows_len)
            elif model_type == 'LSTM':
                #print('LSTM')
                predict_model = model_train_LSTM(save_folder, multi_time_train_t_0_label, multi_time_train_t_0_data, baseline_std_cols, windows_len)
            elif model_type == 'ResNet':
                #print('ResNet')
                predict_model = model_train_ResNet(save_folder, multi_time_train_t_0_label, multi_time_train_t_0_data, baseline_std_cols, windows_len)

            ## 预测准确度
            model_accuracy_baselines(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data)
            ## 预测可信度（方向与幅度）
            #for change_time_len in range(1, windows_len+1): # 改变5个时刻的特征值，change_time_len<=windows_len
            #    model_trustworthiness_all_baselines(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_len, windows_len)
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                model_trustworthiness_all_baselines(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len)
                model_trustworthiness_all_baselines_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len)
        
        elif model_type in ('MDL_DNN', 'MDL_LSTM'):
            if model_type == 'MDL_DNN':
                #print('MDL_DNN')
                predict_model = model_train_MDL_DNN(save_folder, multi_time_train_t_0_label, multi_time_train_t_0_data, baseline_std_cols, windows_len, MDL_baseline_data, multi_time_train_t_0_label_other, multi_time_train_t_0_data_other, MDL_baseline_data_other)
                #print('MDL_DNN 2')
            elif model_type == 'MDL_LSTM':
                predict_model = model_train_MDL_LSTM(save_folder, multi_time_train_t_0_label, multi_time_train_t_0_data, baseline_std_cols, windows_len, MDL_baseline_data, multi_time_train_t_0_label_other, multi_time_train_t_0_data_other, MDL_baseline_data_other)
                        
            #print('预测准确度')
            ## 预测准确度
            model_accuracy_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)
            ## 预测可信度（方向与幅度）
            #for change_time_len in range(1, windows_len+1): # 改变5个时刻的特征值，change_time_len<=windows_len
            #    model_trustworthiness_all_baselines(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_len, windows_len)
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                model_trustworthiness_all_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)
                model_trustworthiness_all_MDL_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len, MDL_baseline_data_test, multi_time_test_t_0_label_other, multi_time_test_t_0_data_other, MDL_baseline_data_test_other)

        elif model_type in ('STCD_MDL_all'):
            if model_type == 'STCD_MDL_all':
                predict_model = model_train_STCD_MDL_DF_TC_loss_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, MDL_model_data_need, MDL_model_data_need_other, SC_loss_weights, TC_loss_weights)
            
            # 精度
            model_accuracy_STCD_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other)
            # 可信度
            multi_time_0_1_n_data = [multi_time_test_t_0_label, multi_time_test_t_0_data, multi_time_test_t_1_label, multi_time_test_t_1_data, multi_time_test_t_n_label, multi_time_test_t_n_data]
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                trustworthiness_all_STCD_MDL(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other)
                trustworthiness_all_STCD_MDL_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, short_label_test, short_data_std_test, long_data_std_test, test_data, feature_diff_ss, change_time_type, windows_len, multi_time_0_1_n_data, temporal_discount_rate_test, accuracy_data_other, MDL_model_data_need_test, MDL_model_data_need_test_other)
        
        elif model_type == 'Linear':
            predict_model = model_train_linear(train_data, model_std_cols, label_next_std)
            ## 预测准确度
            model_accuracy_linear(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, model_std_cols)
            ## 预测可信度（方向与幅度）
            #for change_time_len in range(1, windows_len+1): # 改变5个时刻的特征值，change_time_len<=windows_len
            #    model_trustworthiness_all_baselines(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_len, windows_len)
            for change_time_type in range(1, 3): # 改变1个时刻和全部时刻的特征值，change_time_len<=windows_len
                model_trustworthiness_all_linear(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len, model_std_cols)
                model_trustworthiness_all_linear_PDP(save_folder, model_type, predict_model, label_next_ss, label_diff_ss, multi_time_test_t_0_label, multi_time_test_t_0_data, test_data, feature_ss, change_time_type, windows_len, model_std_cols)
         
        elif model_type == 'Fitted_physical_model':
            fitted_physical_model_prepare(train_data, file_name)
            
            #is_fitted = False
            is_fitted = True
            model_accuracy_physical(save_folder, model_type, label_next_ss, test_data, file_name, is_fitted)
            for change_time_type in range(1, 3):
                model_trustworthiness_all_physical(save_folder, model_type, label_next_ss, test_data, file_name, is_fitted, change_time_type, windows_len)
                model_trustworthiness_all_physical_PDP(save_folder, model_type, label_next_ss, test_data, file_name, is_fitted, change_time_type, windows_len)
            
        

    return 'success'
