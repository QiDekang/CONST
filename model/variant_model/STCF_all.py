from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Subtract
from keras.models import Model
from keras import callbacks, initializers
from keras import regularizers
from tensorflow import optimizers
from common.config import dropout_rate, verbose_level
from model.base_subnetwork.embedding_layers import create_embedding_layers
from model.base_subnetwork.fusion_network import create_fusion_network
from model.base_subnetwork.TC_module import create_TC_module
from model.base_subnetwork.weighted_layer import create_weighted_layer
from model.base_subnetwork.feature_extraction_layer import create_feature_extraction_layer


###################################################
# k+1个时间一致性网络之间不共享参数，仅融合层共享参数
###################################################
def model_build_STCF_all(input_features_size):

    ################################
    # 目标站点input
    ################################
    # t-0时刻input
    input_current_wind_0 = Input(shape=(1,), dtype='int32')
    input_current_weather_0 = Input(shape=(1,), dtype='int32')
    input_current_day_0 = Input(shape=(1,), dtype='int32')
    input_current_hour_0 = Input(shape=(1,), dtype='int32')
    input_current_people_0 = Input(shape=(1,), dtype='int32')

    input_short_0 = Input(shape=(input_features_size,), dtype='float')
    input_short_wind_0 = Input(shape=(1,), dtype='int32')
    input_short_weather_0 = Input(shape=(1,), dtype='int32')
    input_short_day_0 = Input(shape=(1,), dtype='int32')
    input_short_hour_0 = Input(shape=(1,), dtype='int32')
    input_short_people_0 = Input(shape=(1,), dtype='int32')

    input_long_0 = Input(shape=(input_features_size,), dtype='float')
    input_long_wind_0 = Input(shape=(1,), dtype='int32')
    input_long_weather_0 = Input(shape=(1,), dtype='int32')
    input_long_day_0 = Input(shape=(1,), dtype='int32')
    input_long_hour_0 = Input(shape=(1,), dtype='int32')
    input_long_people_0 = Input(shape=(1,), dtype='int32')

    # t-1时刻input
    input_current_wind_1 = Input(shape=(1,), dtype='int32')
    input_current_weather_1 = Input(shape=(1,), dtype='int32')
    input_current_day_1 = Input(shape=(1,), dtype='int32')
    input_current_hour_1 = Input(shape=(1,), dtype='int32')
    input_current_people_1 = Input(shape=(1,), dtype='int32')

    input_short_1 = Input(shape=(input_features_size,), dtype='float')
    input_short_wind_1 = Input(shape=(1,), dtype='int32')
    input_short_weather_1 = Input(shape=(1,), dtype='int32')
    input_short_day_1 = Input(shape=(1,), dtype='int32')
    input_short_hour_1 = Input(shape=(1,), dtype='int32')
    input_short_people_1 = Input(shape=(1,), dtype='int32')

    input_long_1 = Input(shape=(input_features_size,), dtype='float')
    input_long_wind_1 = Input(shape=(1,), dtype='int32')
    input_long_weather_1 = Input(shape=(1,), dtype='int32')
    input_long_day_1 = Input(shape=(1,), dtype='int32')
    input_long_hour_1 = Input(shape=(1,), dtype='int32')
    input_long_people_1 = Input(shape=(1,), dtype='int32')

    # t-2时刻input
    input_current_wind_2 = Input(shape=(1,), dtype='int32')
    input_current_weather_2 = Input(shape=(1,), dtype='int32')
    input_current_day_2 = Input(shape=(1,), dtype='int32')
    input_current_hour_2 = Input(shape=(1,), dtype='int32')
    input_current_people_2 = Input(shape=(1,), dtype='int32')

    input_short_2 = Input(shape=(input_features_size,), dtype='float')
    input_short_wind_2 = Input(shape=(1,), dtype='int32')
    input_short_weather_2 = Input(shape=(1,), dtype='int32')
    input_short_day_2 = Input(shape=(1,), dtype='int32')
    input_short_hour_2 = Input(shape=(1,), dtype='int32')
    input_short_people_2 = Input(shape=(1,), dtype='int32')

    input_long_2 = Input(shape=(input_features_size,), dtype='float')
    input_long_wind_2 = Input(shape=(1,), dtype='int32')
    input_long_weather_2 = Input(shape=(1,), dtype='int32')
    input_long_day_2 = Input(shape=(1,), dtype='int32')
    input_long_hour_2 = Input(shape=(1,), dtype='int32')
    input_long_people_2 = Input(shape=(1,), dtype='int32')


    ################################
    # 其他站点input
    ################################
    # t-0时刻input
    other_stations_input_current_wind_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_weather_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_day_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_hour_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_people_0 = Input(shape=(1,), dtype='int32')

    other_stations_input_short_0 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_short_wind_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_weather_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_day_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_hour_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_people_0 = Input(shape=(1,), dtype='int32')

    other_stations_input_long_0 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_long_wind_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_weather_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_day_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_hour_0 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_people_0 = Input(shape=(1,), dtype='int32')

    # t-1时刻input
    other_stations_input_current_wind_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_weather_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_day_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_hour_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_people_1 = Input(shape=(1,), dtype='int32')

    other_stations_input_short_1 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_short_wind_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_weather_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_day_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_hour_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_people_1 = Input(shape=(1,), dtype='int32')

    other_stations_input_long_1 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_long_wind_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_weather_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_day_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_hour_1 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_people_1 = Input(shape=(1,), dtype='int32')

    # t-2时刻input
    other_stations_input_current_wind_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_weather_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_day_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_hour_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_current_people_2 = Input(shape=(1,), dtype='int32')

    other_stations_input_short_2 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_short_wind_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_weather_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_day_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_hour_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_short_people_2 = Input(shape=(1,), dtype='int32')

    other_stations_input_long_2 = Input(shape=(input_features_size,), dtype='float')
    other_stations_input_long_wind_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_weather_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_day_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_hour_2 = Input(shape=(1,), dtype='int32')
    other_stations_input_long_people_2 = Input(shape=(1,), dtype='int32')




    ################################
    # 模型
    ################################

    ## 不同站点间的特征提取层不共享参数，同一站点处理不同时刻数据时特征提取层共享参数。仅处理连续特征。输出维度需用于后续网络。
    feature_extraction_layer_size = input_features_size # to do, 32时不起作用
    ## 目标站点
    target_station_feature_extraction_layer = create_feature_extraction_layer(input_features_size, feature_extraction_layer_size)
    ## 不同时刻数据特征提取
    input_short_0 = target_station_feature_extraction_layer(input_short_0)
    input_long_0 = target_station_feature_extraction_layer(input_long_0)
    input_short_1 = target_station_feature_extraction_layer(input_short_1)
    input_long_1 = target_station_feature_extraction_layer(input_long_1)
    input_short_2 = target_station_feature_extraction_layer(input_short_2)
    input_long_2 = target_station_feature_extraction_layer(input_long_2)

    ## 其他站点
    other_stations_feature_extraction_layer = create_feature_extraction_layer(input_features_size, feature_extraction_layer_size)
    ## 不同时刻数据特征提取
    other_stations_input_short_0 = other_stations_feature_extraction_layer(other_stations_input_short_0)
    other_stations_input_long_0 = other_stations_feature_extraction_layer(other_stations_input_long_0)
    other_stations_input_short_1 = other_stations_feature_extraction_layer(other_stations_input_short_1)
    other_stations_input_long_1 = other_stations_feature_extraction_layer(other_stations_input_long_1)
    other_stations_input_short_2 = other_stations_feature_extraction_layer(other_stations_input_short_2)
    other_stations_input_long_2 = other_stations_feature_extraction_layer(other_stations_input_long_2)


    ###########################################
    # 时间一致性网络共享参数。各个层仅创建一次即可

    ## 目标站点
    # TC_module不共享参数
    ## 创建网络
    # input_features_size 需要改成 feature_extraction_layer_size
    TC_module_0 = create_TC_module(feature_extraction_layer_size)
    short_prediction_0, long_prediction_0 = TC_module_0([input_current_wind_0, input_current_weather_0, input_current_day_0, input_current_hour_0, input_current_people_0, input_short_0, input_short_wind_0, input_short_weather_0, input_short_day_0, input_short_hour_0, input_short_people_0, input_long_0, input_long_wind_0, input_long_weather_0, input_long_day_0, input_long_hour_0, input_long_people_0])
    ## 创建网络
    TC_module_1 = create_TC_module(feature_extraction_layer_size)
    short_prediction_1, long_prediction_1 = TC_module_1([input_current_wind_1, input_current_weather_1, input_current_day_1, input_current_hour_1, input_current_people_1, input_short_1, input_short_wind_1, input_short_weather_1, input_short_day_1, input_short_hour_1, input_short_people_1, input_long_1, input_long_wind_1, input_long_weather_1, input_long_day_1, input_long_hour_1, input_long_people_1])
    ## 创建网络
    TC_module_2 = create_TC_module(feature_extraction_layer_size)
    short_prediction_2, long_prediction_2 = TC_module_2([input_current_wind_2, input_current_weather_2, input_current_day_2, input_current_hour_2, input_current_people_2, input_short_2, input_short_wind_2, input_short_weather_2, input_short_day_2, input_short_hour_2, input_short_people_2, input_long_2, input_long_wind_2, input_long_weather_2, input_long_day_2, input_long_hour_2, input_long_people_2])
    
    ## 其他站点，不需要再创建网络
    other_stations_short_prediction_0, other_stations_long_prediction_0 = TC_module_0([other_stations_input_current_wind_0, other_stations_input_current_weather_0, other_stations_input_current_day_0, other_stations_input_current_hour_0, other_stations_input_current_people_0, other_stations_input_short_0, other_stations_input_short_wind_0, other_stations_input_short_weather_0, other_stations_input_short_day_0, other_stations_input_short_hour_0, other_stations_input_short_people_0, other_stations_input_long_0, other_stations_input_long_wind_0, other_stations_input_long_weather_0, other_stations_input_long_day_0, other_stations_input_long_hour_0, other_stations_input_long_people_0])
    other_stations_short_prediction_1, other_stations_long_prediction_1 = TC_module_1([other_stations_input_current_wind_1, other_stations_input_current_weather_1, other_stations_input_current_day_1, other_stations_input_current_hour_1, other_stations_input_current_people_1, other_stations_input_short_1, other_stations_input_short_wind_1, other_stations_input_short_weather_1, other_stations_input_short_day_1, other_stations_input_short_hour_1, other_stations_input_short_people_1, other_stations_input_long_1, other_stations_input_long_wind_1, other_stations_input_long_weather_1, other_stations_input_long_day_1, other_stations_input_long_hour_1, other_stations_input_long_people_1])
    other_stations_short_prediction_2, other_stations_long_prediction_2 = TC_module_2([other_stations_input_current_wind_2, other_stations_input_current_weather_2, other_stations_input_current_day_2, other_stations_input_current_hour_2, other_stations_input_current_people_2, other_stations_input_short_2, other_stations_input_short_wind_2, other_stations_input_short_weather_2, other_stations_input_short_day_2, other_stations_input_short_hour_2, other_stations_input_short_people_2, other_stations_input_long_2, other_stations_input_long_wind_2, other_stations_input_long_weather_2, other_stations_input_long_day_2, other_stations_input_long_hour_2, other_stations_input_long_people_2])


    # 拼接
    ## 目标站点
    merged_short_prediction = layers.concatenate([short_prediction_0, short_prediction_1, short_prediction_2], axis=1)  # 从第1个维度拼接
    merged_long_prediction = layers.concatenate([long_prediction_0, long_prediction_1, long_prediction_2], axis=1)  # 从第1个维度拼接
    
    ## 其他站点
    other_stations_merged_short_prediction = layers.concatenate([other_stations_short_prediction_0, other_stations_short_prediction_1, other_stations_short_prediction_2], axis=1)  # 从第1个维度拼接
    other_stations_merged_long_prediction = layers.concatenate([other_stations_long_prediction_0, other_stations_long_prediction_1, other_stations_long_prediction_2], axis=1)  # 从第1个维度拼接
    

    ## 目标站点
    # 多时刻融合层，共享权重
    ## 使用weight_layer
    # 输入维度为3
    merged_size = 3
    weighted_layer = create_weighted_layer(merged_size)
    weighted_short_prediction = weighted_layer(merged_short_prediction)
    weighted_long_prediction = weighted_layer(merged_long_prediction)

    ## 其他站点
    other_stations_weighted_short_prediction = weighted_layer(other_stations_merged_short_prediction)
    other_stations_weighted_long_prediction = weighted_layer(other_stations_merged_long_prediction)


    ##############
    # weighted_short_prediction直接作为空间一致性网络的输出
    ## 四个loss要加起来。

    ### 目标站点
    #  返回自身，并给层命名，以便于loss中操作
    weighted_short_self_layer = Lambda(lambda tensors: tensors, name='weighted_short_prediction_layer')
    weighted_short_prediction = weighted_short_self_layer(weighted_short_prediction)
    weighted_long_self_layer = Lambda(lambda tensors: tensors, name='weighted_long_prediction_layer')
    weighted_long_prediction = weighted_long_self_layer(weighted_long_prediction)
    
    ### 其他站点
    #  返回自身，并给层命名，以便于loss中操作
    other_stations_weighted_short_self_layer = Lambda(lambda tensors: tensors, name='other_stations_weighted_short_prediction_layer')
    other_stations_weighted_short_prediction = other_stations_weighted_short_self_layer(other_stations_weighted_short_prediction)
    other_stations_weighted_long_self_layer = Lambda(lambda tensors: tensors, name='other_stations_weighted_long_prediction_layer')
    other_stations_weighted_long_prediction = other_stations_weighted_long_self_layer(other_stations_weighted_long_prediction)



    ####  模型构建
    # 输入输出
    model = Model(inputs=[input_current_wind_0, input_current_weather_0, input_current_day_0, input_current_hour_0, input_current_people_0, input_short_0, input_short_wind_0, input_short_weather_0, input_short_day_0, input_short_hour_0, input_short_people_0, input_long_0, input_long_wind_0, input_long_weather_0, input_long_day_0, input_long_hour_0, input_long_people_0, input_current_wind_1, input_current_weather_1, input_current_day_1, input_current_hour_1, input_current_people_1, input_short_1, input_short_wind_1, input_short_weather_1, input_short_day_1, input_short_hour_1, input_short_people_1, input_long_1, input_long_wind_1, input_long_weather_1, input_long_day_1, input_long_hour_1, input_long_people_1, input_current_wind_2, input_current_weather_2, input_current_day_2, input_current_hour_2, input_current_people_2, input_short_2, input_short_wind_2, input_short_weather_2, input_short_day_2, input_short_hour_2, input_short_people_2, input_long_2, input_long_wind_2, input_long_weather_2, input_long_day_2, input_long_hour_2, input_long_people_2, other_stations_input_current_wind_0, other_stations_input_current_weather_0, other_stations_input_current_day_0, other_stations_input_current_hour_0, other_stations_input_current_people_0, other_stations_input_short_0, other_stations_input_short_wind_0, other_stations_input_short_weather_0, other_stations_input_short_day_0, other_stations_input_short_hour_0, other_stations_input_short_people_0, other_stations_input_long_0, other_stations_input_long_wind_0, other_stations_input_long_weather_0, other_stations_input_long_day_0, other_stations_input_long_hour_0, other_stations_input_long_people_0, other_stations_input_current_wind_1, other_stations_input_current_weather_1, other_stations_input_current_day_1, other_stations_input_current_hour_1, other_stations_input_current_people_1, other_stations_input_short_1, other_stations_input_short_wind_1, other_stations_input_short_weather_1, other_stations_input_short_day_1, other_stations_input_short_hour_1, other_stations_input_short_people_1, other_stations_input_long_1, other_stations_input_long_wind_1, other_stations_input_long_weather_1, other_stations_input_long_day_1, other_stations_input_long_hour_1, other_stations_input_long_people_1, other_stations_input_current_wind_2, other_stations_input_current_weather_2, other_stations_input_current_day_2, other_stations_input_current_hour_2, other_stations_input_current_people_2, other_stations_input_short_2, other_stations_input_short_wind_2, other_stations_input_short_weather_2, other_stations_input_short_day_2, other_stations_input_short_hour_2, other_stations_input_short_people_2, other_stations_input_long_2, other_stations_input_long_wind_2, other_stations_input_long_weather_2, other_stations_input_long_day_2, other_stations_input_long_hour_2, other_stations_input_long_people_2], outputs=[weighted_short_prediction, weighted_long_prediction, other_stations_weighted_short_prediction, other_stations_weighted_long_prediction])
    # compile
    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss={
                      'weighted_short_prediction_layer': 'mae',
                      'weighted_long_prediction_layer': 'mae',
                      'other_stations_weighted_short_prediction_layer': 'mae',
                      'other_stations_weighted_long_prediction_layer': 'mae'
                  },
                  loss_weights={
                      'weighted_short_prediction_layer': 1.,
                      'weighted_long_prediction_layer': 1.,
                      'other_stations_weighted_short_prediction_layer': 1.,
                      'other_stations_weighted_long_prediction_layer': 1.
                  },
                  metrics=['mae', 'mse', 'mape'])
    
    return model


def model_train_STCF_all(save_path, MMF_data_set, other_stations_MMF_data_set, feature_cols, label_diff):

    train_current_0, train_short_0, train_long_0, train_current_1, train_short_1, train_long_1, train_current_2, train_short_2, train_long_2, test_current_0, test_short_0, test_long_0, test_current_1, test_short_1, test_long_1, test_current_2, test_short_2, test_long_2 = MMF_data_set
    #print('train_current_0', train_current_0)

    # 其他站点数据
    other_stations_train_current_0, other_stations_train_short_0, other_stations_train_long_0, other_stations_train_current_1, other_stations_train_short_1, other_stations_train_long_1, other_stations_train_current_2, other_stations_train_short_2, other_stations_train_long_2, other_stations_test_current_0, other_stations_test_short_0, other_stations_test_long_0, other_stations_test_current_1, other_stations_test_short_1, other_stations_test_long_1, other_stations_test_current_2, other_stations_test_short_2, other_stations_test_long_2 = other_stations_MMF_data_set


    input_features_size = len(feature_cols)

    # model_build
    model = model_build_STCF_all(input_features_size)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_path + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # model.fit
    model.fit([train_current_0["wind_direction"], train_current_0["weather"], train_current_0["day"], train_current_0["hour"], train_current_0["havePeople"], train_short_0[feature_cols], train_short_0["wind_direction"], train_short_0["weather"], train_short_0["day"], train_short_0["hour"], train_short_0["havePeople"], train_long_0[feature_cols], train_long_0["wind_direction"], train_long_0["weather"], train_long_0["day"], train_long_0["hour"], train_long_0["havePeople"], train_current_1["wind_direction"], train_current_1["weather"], train_current_1["day"], train_current_1["hour"], train_current_1["havePeople"], train_short_1[feature_cols], train_short_1["wind_direction"], train_short_1["weather"], train_short_1["day"], train_short_1["hour"], train_short_1["havePeople"], train_long_1[feature_cols], train_long_1["wind_direction"], train_long_1["weather"], train_long_1["day"], train_long_1["hour"], train_long_1["havePeople"], train_current_2["wind_direction"], train_current_2["weather"], train_current_2["day"], train_current_2["hour"], train_current_2["havePeople"], train_short_2[feature_cols], train_short_2["wind_direction"], train_short_2["weather"], train_short_2["day"], train_short_2["hour"], train_short_2["havePeople"], train_long_2[feature_cols], train_long_2["wind_direction"], train_long_2["weather"], train_long_2["day"], train_long_2["hour"], train_long_2["havePeople"], other_stations_train_current_0["wind_direction"], other_stations_train_current_0["weather"], other_stations_train_current_0["day"], other_stations_train_current_0["hour"], other_stations_train_current_0["havePeople"], other_stations_train_short_0[feature_cols], other_stations_train_short_0["wind_direction"], other_stations_train_short_0["weather"], other_stations_train_short_0["day"], other_stations_train_short_0["hour"], other_stations_train_short_0["havePeople"], other_stations_train_long_0[feature_cols], other_stations_train_long_0["wind_direction"], other_stations_train_long_0["weather"], other_stations_train_long_0["day"], other_stations_train_long_0["hour"], other_stations_train_long_0["havePeople"], other_stations_train_current_1["wind_direction"], other_stations_train_current_1["weather"], other_stations_train_current_1["day"], other_stations_train_current_1["hour"], other_stations_train_current_1["havePeople"], other_stations_train_short_1[feature_cols], other_stations_train_short_1["wind_direction"], other_stations_train_short_1["weather"], other_stations_train_short_1["day"], other_stations_train_short_1["hour"], other_stations_train_short_1["havePeople"], other_stations_train_long_1[feature_cols], other_stations_train_long_1["wind_direction"], other_stations_train_long_1["weather"], other_stations_train_long_1["day"], other_stations_train_long_1["hour"], other_stations_train_long_1["havePeople"], other_stations_train_current_2["wind_direction"], other_stations_train_current_2["weather"], other_stations_train_current_2["day"], other_stations_train_current_2["hour"], other_stations_train_current_2["havePeople"], other_stations_train_short_2[feature_cols], other_stations_train_short_2["wind_direction"], other_stations_train_short_2["weather"], other_stations_train_short_2["day"], other_stations_train_short_2["hour"], other_stations_train_short_2["havePeople"], other_stations_train_long_2[feature_cols], other_stations_train_long_2["wind_direction"], other_stations_train_long_2["weather"], other_stations_train_long_2["day"], other_stations_train_long_2["hour"], other_stations_train_long_2["havePeople"]],
              [train_short_0[label_diff], train_long_0[label_diff], other_stations_train_short_0[label_diff], other_stations_train_long_0[label_diff]],
              epochs=300,  # 避免过拟合，减小循环次数
              batch_size=32,
              #callbacks=[save_best, early_stop],
              callbacks=[early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
