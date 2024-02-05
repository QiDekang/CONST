from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Subtract, LSTM, Add, Multiply
from keras.models import Model
from keras import callbacks, initializers
from keras import regularizers
from tensorflow import optimizers
from common.config import dropout_rate, verbose_level
from common.config import embed_size, discrete_feature_num, learning_rate
from STCD_models.modules.embedding_layers import create_embedding_layers
from STCD_models.modules.fusion_layers import create_fusion_layers
from STCD_models.modules.LSTM_fusion_layers import create_LSTM_fusion_layers
import numpy as np
from STCD_models.modules.LSTM_fusion_layers_sub_1 import create_LSTM_fusion_layers_sub_1
from STCD_models.modules.LSTM_fusion_layers_sub_2 import create_LSTM_fusion_layers_sub_2
#from STCD_models.modules.MDL_fusion_layers_sub_1 import create_MDL_fusion_layers_sub_1 #基于LSTM的ResNet
#from STCD_models.modules.MDL_fusion_layers_sub_2 import create_MDL_fusion_layers_sub_2
from STCD_models.modules.MDL_DNN_fusion_layers_sub_1 import create_MDL_fusion_layers_sub_1 #基于DNN的ResNet
from STCD_models.modules.MDL_DNN_fusion_layers_sub_2 import create_MDL_fusion_layers_sub_2 #基于DNN的ResNet
from STCD_models.modules.discrete_fusion_sub_1 import create_discrete_fusion_layers_sub_1
from STCD_models.modules.discrete_fusion_sub_2 import create_discrete_fusion_layers_sub_2
from STCD_models.modules.weighted_layer import create_weighted_layer


################
################
def model_build_STCD_MDL_DF_TC_loss_SC(input_features_size, windows_len, SC_loss_weights, TC_loss_weights):

    # MDL 数据
    continuous_model_day_short_diff_std = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_day_long_diff_std = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_week_short_diff_std = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_week_long_diff_std = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_day_short_diff_std_other = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_day_long_diff_std_other = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_week_short_diff_std_other = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_model_week_long_diff_std_other = Input(shape=(windows_len, input_features_size), dtype='float')



    continuous_data_short_diff_all_std = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_data_long_diff_all_std = Input(shape=(windows_len, input_features_size), dtype='float')

    # other
    continuous_data_short_diff_all_std_other = Input(shape=(windows_len, input_features_size), dtype='float')
    continuous_data_long_diff_all_std_other = Input(shape=(windows_len, input_features_size), dtype='float')

    # 循环生成input
    for index in range(0, 3):
        if index == 2:
            index_name = 'n'
        else:
            index_name = str(index)
        globals()['wind_data_t_' + index_name] = Input(shape=(windows_len,), dtype='int32')
        globals()['weather_data_t_' + index_name] = Input(shape=(windows_len,), dtype='int32')
        globals()['day_data_t_' + index_name] = Input(shape=(windows_len,), dtype='int32')
        globals()['hour_data_t_' + index_name] = Input(shape=(windows_len,), dtype='int32')
        globals()['havePeople_data_t_' + index_name] = Input(shape=(windows_len,), dtype='int32')

    # other station
    for index in range(0, 3):
        if index == 2:
            index_name = 'n'
        else:
            index_name = str(index)
        globals()['wind_data_t_' + index_name + '_other'] = Input(shape=(windows_len,), dtype='int32')
        globals()['weather_data_t_' + index_name + '_other'] = Input(shape=(windows_len,), dtype='int32')
        globals()['day_data_t_' + index_name + '_other'] = Input(shape=(windows_len,), dtype='int32')
        globals()['hour_data_t_' + index_name + '_other'] = Input(shape=(windows_len,), dtype='int32')
        globals()['havePeople_data_t_' + index_name + '_other'] = Input(shape=(windows_len,), dtype='int32')

    # 预处理。使用离散特征嵌入层
    ## embedding_layers
    #discrete_feature_num = 5  #离散特征数量
    #embed_size = 2  #离散特征嵌入后输出的维度。
    embedding_layers, embedded_features_size = create_embedding_layers(windows_len)
    ##  embedding
    embedding_t_0 = embedding_layers([wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0])
    embedding_t_1 = embedding_layers([wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1])
    embedding_t_n = embedding_layers([wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n])

    # other station
    embedding_t_0_other = embedding_layers([wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other])
    embedding_t_1_other = embedding_layers([wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other])
    embedding_t_n_other = embedding_layers([wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other])

    
    # 离散特征差分模块
    subtract_feature_short = Subtract(name='short_diff')([embedding_t_0, embedding_t_1])
    subtract_feature_long = Subtract(name='long_diff')([embedding_t_0, embedding_t_n])

    # other station
    subtract_feature_short_other = Subtract(name='short_diff_other')([embedding_t_0_other, embedding_t_1_other])
    subtract_feature_long_other = Subtract(name='long_diff_other')([embedding_t_0_other, embedding_t_n_other])


    ## 拼接连续特征和离散特征
    #merged_short = layers.concatenate([continuous_data_short_diff_all_std, subtract_feature_short], axis=2)  # 从第2个维度拼接
    #merged_long = layers.concatenate([continuous_data_long_diff_all_std, subtract_feature_long], axis=2)  # 从第2个维度拼接

    # other station
    #merged_short_other = layers.concatenate([continuous_data_short_diff_all_std_other, subtract_feature_short_other], axis=2)  # 从第2个维度拼接
    #merged_long_other = layers.concatenate([continuous_data_long_diff_all_std_other, subtract_feature_long_other], axis=2)  # 从第2个维度拼接

    # 简单处理，都拼接
    #merged_short_all = layers.concatenate([merged_short, continuous_model_day_short_diff_std, continuous_model_week_short_diff_std], axis=2)  # 从第2个维度拼接
    #merged_long_all = layers.concatenate([merged_short, continuous_model_day_long_diff_std, continuous_model_week_long_diff_std], axis=2)  # 从第2个维度拼接
    #merged_short_all_other = layers.concatenate([merged_short_other, continuous_model_day_short_diff_std_other, continuous_model_week_short_diff_std_other], axis=2)  # 从第2个维度拼接
    #merged_long_all_other = layers.concatenate([merged_long_other, continuous_model_day_long_diff_std, continuous_model_week_long_diff_std_other], axis=2)  # 从第2个维度拼接
    
    # 分四个模块
    ## 临近、周期、趋势模块
    ### 短期差分和超长期差分上下层全部共享
    ### 多个站点间仅底层共享，上层不共享
    #### 临近数据
    ##### target station
    fusion_network = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    short_prediction = fusion_network(continuous_data_short_diff_all_std)
    long_prediction = fusion_network(continuous_data_long_diff_all_std)
    ##### other station
    short_other_prediction = fusion_network(continuous_data_short_diff_all_std_other)
    long_other_prediction = fusion_network(continuous_data_long_diff_all_std_other)

    #### 周期数据
    ##### target station
    fusion_network_day = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    short_day_prediction = fusion_network_day(continuous_model_day_short_diff_std)
    long_day_prediction = fusion_network_day(continuous_model_day_long_diff_std)
    ##### other station
    short_day_other_prediction = fusion_network_day(continuous_model_day_short_diff_std_other)
    long_day_other_prediction = fusion_network_day(continuous_model_day_long_diff_std_other)

    #### 趋势数据
    ##### target station
    fusion_network_week = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    short_week_prediction = fusion_network_week(continuous_model_week_short_diff_std)
    long_week_prediction = fusion_network_week(continuous_model_week_long_diff_std)
    ##### other station
    short_week_other_prediction = fusion_network_week(continuous_model_week_short_diff_std_other)
    long_week_other_prediction = fusion_network_week(continuous_model_week_long_diff_std_other)


    #### 离散数据
    discrete_fusion_layers = create_discrete_fusion_layers_sub_1(embedded_features_size, windows_len)
    discrete_short_prediction = discrete_fusion_layers(subtract_feature_short)
    discrete_long_prediction = discrete_fusion_layers(subtract_feature_long)
    ##### other station
    discrete_short_other_prediction = discrete_fusion_layers(subtract_feature_short_other)
    discrete_long_other_prediction = discrete_fusion_layers(subtract_feature_long_other)  



    ### 多个站点间上层不共享，针对other station重新创建网络。长短期依然共享
    sub_2_input_size = 32
    #### 临近数据
    ##### target station
    fusion_network_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_prediction_2 = fusion_network_2(short_prediction)
    long_prediction_2 = fusion_network_2(long_prediction)
    ##### other station
    fusion_network_other_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_other_prediction_2 = fusion_network_other_2(short_other_prediction)
    long_other_prediction_2 = fusion_network_other_2(long_other_prediction)

    #### 周期数据
    ##### target station
    fusion_network_day_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_day_prediction_2 = fusion_network_day_2(short_day_prediction)
    long_day_prediction_2 = fusion_network_day_2(long_day_prediction)
    ##### other station
    fusion_network_day_other_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_day_other_prediction_2 = fusion_network_day_other_2(short_day_other_prediction)
    long_day_other_prediction_2 = fusion_network_day_other_2(long_day_other_prediction)

    #### 趋势数据
    ##### target station
    fusion_network_week_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_week_prediction_2 = fusion_network_week_2(short_week_prediction)
    long_week_prediction_2 = fusion_network_week_2(long_week_prediction)
    ##### other station
    fusion_network_week_other_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    short_week_other_prediction_2 = fusion_network_week_other_2(short_week_other_prediction)
    long_week_other_prediction_2 = fusion_network_week_other_2(long_week_other_prediction)


    #### 离散数据
    discrete_fusion_layers_2 = create_discrete_fusion_layers_sub_2(sub_2_input_size, windows_len)
    discrete_short_prediction_2 = discrete_fusion_layers_2(discrete_short_prediction)
    discrete_long_prediction_2 = discrete_fusion_layers_2(discrete_long_prediction)
    ##### other station
    discrete_fusion_layers_other_2 = create_discrete_fusion_layers_sub_2(sub_2_input_size, windows_len)
    discrete_short_other_prediction_2 = discrete_fusion_layers_other_2(discrete_short_other_prediction)
    discrete_long_other_prediction_2 = discrete_fusion_layers_other_2(discrete_long_other_prediction)  


    # 拼接
    merged_short = layers.concatenate([short_prediction_2, short_day_prediction_2, short_week_prediction_2, discrete_short_prediction_2], axis=1)  # 从第2个维度拼接
    merged_long = layers.concatenate([long_prediction_2, long_day_prediction_2, long_week_prediction_2, discrete_long_prediction_2], axis=1)  # 从第2个维度拼接
    
    merged_short_other = layers.concatenate([short_other_prediction_2, short_day_other_prediction_2, short_week_other_prediction_2, discrete_short_other_prediction_2], axis=1)  # 从第2个维度拼接
    merged_long_other = layers.concatenate([long_other_prediction_2, long_day_other_prediction_2, long_week_other_prediction_2, discrete_long_other_prediction_2], axis=1)  # 从第2个维度拼接
    


    # 四个分支融合层
    ## 长短期间共享，站点间不共享
    weight_layer = create_weighted_layer(4)
    short_output = weight_layer(merged_short)
    long_output = weight_layer(merged_long)
    weight_layer_other = create_weighted_layer(4)
    short_other_output = weight_layer_other(merged_short_other)
    long_other_output = weight_layer_other(merged_long_other)




    ## temporal_discount_rank_loss
    temporal_discount_rate = Input(shape=(1,), dtype='float', name='temporal_discount_rate')
    prediction_long_discount = Multiply()([long_output, temporal_discount_rate])

    # other
    temporal_discount_rate_other = Input(shape=(1,), dtype='float', name='temporal_discount_rate_other')
    prediction_long_discount_other = Multiply()([long_other_output, temporal_discount_rate_other])
    


    #  返回自身，并给层命名，以便于loss中操作
    short_self_layer = Lambda(lambda tensors: tensors, name='short_self_layer')
    prediction_short = short_self_layer(short_output)
    long_self_layer = Lambda(lambda tensors: tensors, name='long_self_layer')
    prediction_long_discount = long_self_layer(prediction_long_discount)
    #prediction_long = long_self_layer(prediction_long)

    # other
    short_self_layer_other = Lambda(lambda tensors: tensors, name='short_self_layer_other')
    prediction_short_other = short_self_layer_other(short_other_output)
    long_self_layer_other = Lambda(lambda tensors: tensors, name='long_self_layer_other')
    prediction_long_discount_other = long_self_layer_other(prediction_long_discount_other)
    
    TC_SC_loss_weight = TC_loss_weights * SC_loss_weights

    #  模型构建
    ## 输入输出
    model = Model(inputs=[temporal_discount_rate, continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n, temporal_discount_rate_other, continuous_data_short_diff_all_std_other, continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other, continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std, continuous_model_day_short_diff_std_other, continuous_model_day_long_diff_std_other, continuous_model_week_short_diff_std_other, continuous_model_week_long_diff_std_other], outputs=[prediction_short, prediction_long_discount, prediction_short_other, prediction_long_discount_other])
    ## compile
    opt = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss={
                      'short_self_layer': 'mae',
                      'long_self_layer': 'mae',
                      'short_self_layer_other': 'mae',
                      'long_self_layer_other': 'mae'
                  },
                  loss_weights={
                      'short_self_layer': 1.,
                      'long_self_layer': TC_loss_weights,
                      'short_self_layer_other': SC_loss_weights,
                      'long_self_layer_other': TC_SC_loss_weight
                  },
                  metrics=['mae', 'mse', 'mape'])
    
    return model


def model_train_STCD_MDL_DF_TC_loss_SC(save_folder, label_short_diff_std, label_long_diff_std, short_data_std, long_data_std, model_std_cols, windows_len, temporal_discount_rate, model_train_data_other, MDL_model_data_need, MDL_model_data_need_other, SC_loss_weights, TC_loss_weights):

    input_features_size = len(model_std_cols)

    continuous_data_short_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1 = short_data_std
    continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n = long_data_std

    # 其他站点数据
    label_short_diff_std_other, label_long_diff_std_other, short_data_std_other, long_data_std_other, temporal_discount_rate_other = model_train_data_other
    continuous_data_short_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other = short_data_std_other
    continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other = long_data_std_other



    # MDL数据
    continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std = MDL_model_data_need
    continuous_model_day_short_diff_std_other, continuous_model_day_long_diff_std_other, continuous_model_week_short_diff_std_other, continuous_model_week_long_diff_std_other = MDL_model_data_need_other



    # model_build
    model = model_build_STCD_MDL_DF_TC_loss_SC(input_features_size, windows_len, SC_loss_weights, TC_loss_weights)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_folder + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # 将真实的pairwise_label_i_j乘以temporal_discount_rate
    #print('pairwise_label_i_j_std_data', pairwise_label_i_j_std_data)
    #print('label_long_diff_std', label_long_diff_std)
    label_long_diff_std_discount = np.multiply(label_long_diff_std, temporal_discount_rate)
    label_long_diff_std_discount_other = np.multiply(label_long_diff_std_other, temporal_discount_rate_other)
    #print('label_long_diff_std_discount', label_long_diff_std_discount)


    # model.fit
    model.fit([temporal_discount_rate, continuous_data_short_diff_all_std, continuous_data_long_diff_all_std, wind_data_t_0, weather_data_t_0, day_data_t_0, hour_data_t_0, havePeople_data_t_0, wind_data_t_1, weather_data_t_1, day_data_t_1, hour_data_t_1, havePeople_data_t_1, wind_data_t_n, weather_data_t_n, day_data_t_n, hour_data_t_n, havePeople_data_t_n, temporal_discount_rate_other, continuous_data_short_diff_all_std_other, continuous_data_long_diff_all_std_other, wind_data_t_0_other, weather_data_t_0_other, day_data_t_0_other, hour_data_t_0_other, havePeople_data_t_0_other, wind_data_t_1_other, weather_data_t_1_other, day_data_t_1_other, hour_data_t_1_other, havePeople_data_t_1_other, wind_data_t_n_other, weather_data_t_n_other, day_data_t_n_other, hour_data_t_n_other, havePeople_data_t_n_other, continuous_model_day_short_diff_std, continuous_model_day_long_diff_std, continuous_model_week_short_diff_std, continuous_model_week_long_diff_std, continuous_model_day_short_diff_std_other, continuous_model_day_long_diff_std_other, continuous_model_week_short_diff_std_other, continuous_model_week_long_diff_std_other],
              [label_short_diff_std, label_long_diff_std_discount, label_short_diff_std_other, label_long_diff_std_discount_other],
              epochs=300,  # 避免过拟合，减小循环次数
              #epochs=100,  # 避免过拟合，减小循环次数
              #batch_size=1280,
              batch_size=64,
              callbacks=[save_best, early_stop],
              #callbacks=[early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
