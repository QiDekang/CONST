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


###################################################
# k+1个时间一致性网络之间不共享参数，仅融合层共享参数
###################################################
def model_build_STCF_MFF_D_TC_MMF(input_features_size):

    ################################
    # input
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
    # 模型
    ################################
    # TC_module不共享参数
    ## 创建网络
    TC_module_0 = create_TC_module(input_features_size)
    short_prediction_0, long_prediction_0 = TC_module_0([input_current_wind_0, input_current_weather_0, input_current_day_0, input_current_hour_0, input_current_people_0, input_short_0, input_short_wind_0, input_short_weather_0, input_short_day_0, input_short_hour_0, input_short_people_0, input_long_0, input_long_wind_0, input_long_weather_0, input_long_day_0, input_long_hour_0, input_long_people_0])
    ## 创建网络
    TC_module_1 = create_TC_module(input_features_size)
    short_prediction_1, long_prediction_1 = TC_module_1([input_current_wind_1, input_current_weather_1, input_current_day_1, input_current_hour_1, input_current_people_1, input_short_1, input_short_wind_1, input_short_weather_1, input_short_day_1, input_short_hour_1, input_short_people_1, input_long_1, input_long_wind_1, input_long_weather_1, input_long_day_1, input_long_hour_1, input_long_people_1])
    ## 创建网络
    TC_module_2 = create_TC_module(input_features_size)
    short_prediction_2, long_prediction_2 = TC_module_2([input_current_wind_2, input_current_weather_2, input_current_day_2, input_current_hour_2, input_current_people_2, input_short_2, input_short_wind_2, input_short_weather_2, input_short_day_2, input_short_hour_2, input_short_people_2, input_long_2, input_long_wind_2, input_long_weather_2, input_long_day_2, input_long_hour_2, input_long_people_2])


    # 拼接
    merged_short_prediction = layers.concatenate([short_prediction_0, short_prediction_1, short_prediction_2], axis=1)  # 从第1个维度拼接
    merged_long_prediction = layers.concatenate([long_prediction_0, long_prediction_1, long_prediction_2], axis=1)  # 从第1个维度拼接


    # 融合层，共享权重
    ## 使用weight_layer
    # 输入维度为3
    merged_size = 3
    weighted_layer = create_weighted_layer(merged_size)
    weighted_short_prediction = weighted_layer(merged_short_prediction)
    weighted_long_prediction = weighted_layer(merged_long_prediction)

    #  返回自身，并给层命名，以便于loss中操作
    weighted_short_self_layer = Lambda(lambda tensors: tensors, name='weighted_short_prediction_layer')
    weighted_short_prediction = weighted_short_self_layer(weighted_short_prediction)
    weighted_long_self_layer = Lambda(lambda tensors: tensors, name='weighted_long_prediction_layer')
    weighted_long_prediction = weighted_long_self_layer(weighted_long_prediction)



    ####  模型构建
    # 输入输出
    model = Model(inputs=[input_current_wind_0, input_current_weather_0, input_current_day_0, input_current_hour_0, input_current_people_0, input_short_0, input_short_wind_0, input_short_weather_0, input_short_day_0, input_short_hour_0, input_short_people_0, input_long_0, input_long_wind_0, input_long_weather_0, input_long_day_0, input_long_hour_0, input_long_people_0,
                          input_current_wind_1, input_current_weather_1, input_current_day_1, input_current_hour_1, input_current_people_1, input_short_1, input_short_wind_1, input_short_weather_1, input_short_day_1, input_short_hour_1, input_short_people_1, input_long_1, input_long_wind_1, input_long_weather_1, input_long_day_1, input_long_hour_1, input_long_people_1,
                          input_current_wind_2, input_current_weather_2, input_current_day_2, input_current_hour_2, input_current_people_2, input_short_2, input_short_wind_2, input_short_weather_2, input_short_day_2, input_short_hour_2, input_short_people_2, input_long_2, input_long_wind_2, input_long_weather_2, input_long_day_2, input_long_hour_2, input_long_people_2], outputs=[weighted_short_prediction, weighted_long_prediction])
    # compile
    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss={
                      'weighted_short_prediction_layer': 'mae',
                      'weighted_long_prediction_layer': 'mae'
                  },
                  loss_weights={
                      'weighted_short_prediction_layer': 1.,
                      'weighted_long_prediction_layer': 1.
                  },
                  metrics=['mae', 'mse', 'mape'])
    
    return model


def model_train_STCF_MFF_D_TC_MMF(save_path, MMF_data_set, feature_cols, label_diff):

    train_current_0, train_short_0, train_long_0, train_current_1, train_short_1, train_long_1, train_current_2, train_short_2, train_long_2, test_current_0, test_short_0, test_long_0, test_current_1, test_short_1, test_long_1, test_current_2, test_short_2, test_long_2 = MMF_data_set
    #print('train_current_0', train_current_0)


    input_features_size = len(feature_cols)

    # model_build
    model = model_build_STCF_MFF_D_TC_MMF(input_features_size)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_path + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # model.fit
    model.fit([train_current_0["wind_direction"], train_current_0["weather"], train_current_0["day"], train_current_0["hour"], train_current_0["havePeople"], train_short_0[feature_cols], train_short_0["wind_direction"], train_short_0["weather"], train_short_0["day"], train_short_0["hour"], train_short_0["havePeople"], train_long_0[feature_cols], train_long_0["wind_direction"], train_long_0["weather"], train_long_0["day"], train_long_0["hour"], train_long_0["havePeople"], train_current_1["wind_direction"], train_current_1["weather"], train_current_1["day"], train_current_1["hour"], train_current_1["havePeople"], train_short_1[feature_cols], train_short_1["wind_direction"], train_short_1["weather"], train_short_1["day"], train_short_1["hour"], train_short_1["havePeople"], train_long_1[feature_cols], train_long_1["wind_direction"], train_long_1["weather"], train_long_1["day"], train_long_1["hour"], train_long_1["havePeople"], train_current_2["wind_direction"], train_current_2["weather"], train_current_2["day"], train_current_2["hour"], train_current_2["havePeople"], train_short_2[feature_cols], train_short_2["wind_direction"], train_short_2["weather"], train_short_2["day"], train_short_2["hour"], train_short_2["havePeople"], train_long_2[feature_cols], train_long_2["wind_direction"], train_long_2["weather"], train_long_2["day"], train_long_2["hour"], train_long_2["havePeople"]],
              [train_short_0[label_diff], train_long_0[label_diff]],
              epochs=300,  # 避免过拟合，减小循环次数
              batch_size=32,
              #callbacks=[save_best, early_stop],
              callbacks=[early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
