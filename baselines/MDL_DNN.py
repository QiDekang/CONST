from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Subtract
from keras.models import Model
from keras import callbacks, initializers
from keras import regularizers
from tensorflow import optimizers
from common.config import dropout_rate, verbose_level
from common.config import embed_size, discrete_feature_num, learning_rate
from STCD_models.modules.embedding_layers import create_embedding_layers
from STCD_models.modules.fusion_layers import create_fusion_layers
from STCD_models.modules.fusion_layers_deep import create_fusion_layers_deep
from STCD_models.modules.MDL_fusion_layers import create_MDL_fusion_layers
from STCD_models.modules.MDL_DNN_fusion_layers_sub_1 import create_MDL_fusion_layers_sub_1
from STCD_models.modules.MDL_DNN_fusion_layers_sub_2 import create_MDL_fusion_layers_sub_2
from STCD_models.modules.weighted_layer import create_weighted_layer
from STCD_models.modules.discrete_fusion import create_discrete_fusion_layers
from STCD_models.modules.discrete_fusion_sub_1 import create_discrete_fusion_layers_sub_1
from STCD_models.modules.discrete_fusion_sub_2 import create_discrete_fusion_layers_sub_2


################
# DNN with indoor temperature
# 输入为当前时刻的连续特征（实际值）以及离散特征。输出为t+1时刻的室温。
################
def model_build_MDL(input_features_size, windows_len):

    # input
    continuous_baseline_data = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_data') # 当前时刻连续特征的实际值
    continuous_baseline_day_data = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_day_data') # 当前时刻连续特征的实际值
    continuous_baseline_week_data = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_week_data') # 当前时刻连续特征的实际值
    wind_data = Input(shape=(windows_len,), dtype='int32', name='wind_data')
    weather_data = Input(shape=(windows_len,), dtype='int32', name='weather_data')
    day_data = Input(shape=(windows_len,), dtype='int32', name='day_data')
    hour_data = Input(shape=(windows_len,), dtype='int32', name='hour_data')
    havePeople_data = Input(shape=(windows_len,), dtype='int32', name='havePeople_data')

    continuous_baseline_data_other = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_data_other') # 当前时刻连续特征的实际值
    continuous_baseline_day_data_other = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_day_data_other') # 当前时刻连续特征的实际值
    continuous_baseline_week_data_other = Input(shape=(windows_len, input_features_size), dtype='float', name='continuous_baseline_week_data_other') # 当前时刻连续特征的实际值
    wind_data_other = Input(shape=(windows_len,), dtype='int32', name='wind_data_other')
    weather_data_other = Input(shape=(windows_len,), dtype='int32', name='weather_data_other')
    day_data_other = Input(shape=(windows_len,), dtype='int32', name='day_data_other')
    hour_data_other = Input(shape=(windows_len,), dtype='int32', name='hour_data_other')
    havePeople_data_other = Input(shape=(windows_len,), dtype='int32', name='havePeople_data_other')

    # 预处理。使用离散特征嵌入层
    ## embedding_layers
    #discrete_feature_num = 5  #离散特征数量
    #embed_size = 2  #离散特征嵌入后输出的维度。
    embedding_layers, embedded_features_size = create_embedding_layers(windows_len)
    ##  embedding
    embedding_current = embedding_layers([wind_data, weather_data, day_data, hour_data, havePeople_data])
    embedding_other = embedding_layers([wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other])
    #print(embedding_current)
    #print(input_current)
    ## 拼接连续特征和离散特征
    #merged_current_features = layers.concatenate([continuous_baseline_data, embedding_current], axis=2)  # 从第2个维度拼接
    #merged_current_features_other = layers.concatenate([continuous_baseline_data_other, embedding_other], axis=2)  # 从第2个维度拼接

    # 融合网络
    ## 创建网络
    #fusion_feature_size = input_features_size + embedded_features_size
    #fusion_network = create_fusion_layers(fusion_feature_size, windows_len)
    # 三个独立的网络
    ## 底层共享
    fusion_network = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    continuous_baseline_data_prediction = fusion_network(continuous_baseline_data)
    continuous_baseline_data_other_prediction = fusion_network(continuous_baseline_data_other)

    fusion_network_day = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    continuous_baseline_day_data_prediction = fusion_network_day(continuous_baseline_day_data)
    continuous_baseline_day_data_other_prediction = fusion_network_day(continuous_baseline_day_data_other)

    fusion_network_week = create_MDL_fusion_layers_sub_1(input_features_size, windows_len)
    continuous_baseline_week_data_prediction = fusion_network_week(continuous_baseline_week_data)
    continuous_baseline_week_data_other_prediction = fusion_network_week(continuous_baseline_week_data_other)

    # 离散特征网络
    discrete_fusion_layers = create_discrete_fusion_layers_sub_1(embedded_features_size, windows_len)
    discrete_prediction = discrete_fusion_layers(embedding_current)
    discrete_other_prediction = discrete_fusion_layers(embedding_other)

    ## 顶层不共享
    sub_2_input_size = 32
    fusion_network_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    continuous_baseline_data_prediction_2 = fusion_network_2(continuous_baseline_data_prediction)
    continuous_baseline_data_other_prediction_2 = fusion_network_2(continuous_baseline_data_other_prediction)

    fusion_network_day_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    continuous_baseline_day_data_prediction_2 = fusion_network_day_2(continuous_baseline_day_data_prediction)
    continuous_baseline_day_data_other_prediction_2 = fusion_network_day_2(continuous_baseline_day_data_other_prediction)

    fusion_network_week_2 = create_MDL_fusion_layers_sub_2(sub_2_input_size, windows_len)
    continuous_baseline_week_data_prediction_2 = fusion_network_week_2(continuous_baseline_week_data_prediction)
    continuous_baseline_week_data_other_prediction_2 = fusion_network_week_2(continuous_baseline_week_data_other_prediction)

    # 离散特征网络
    discrete_fusion_layers_2 = create_discrete_fusion_layers_sub_2(sub_2_input_size, windows_len)
    discrete_prediction_2 = discrete_fusion_layers_2(discrete_prediction)
    discrete_other_prediction_2 = discrete_fusion_layers_2(discrete_other_prediction)



    merged_output = layers.concatenate([continuous_baseline_data_prediction_2, continuous_baseline_day_data_prediction_2, continuous_baseline_week_data_prediction_2, discrete_prediction_2], axis=1)  # 从第2个维度拼接
    merged_output_other = layers.concatenate([continuous_baseline_data_other_prediction_2, continuous_baseline_day_data_other_prediction_2, continuous_baseline_week_data_other_prediction_2, discrete_other_prediction_2], axis=1)  # 从第2个维度拼接
    
    # 四个分支融合层，不共享
    weight_layer = create_weighted_layer(4)
    final_output = weight_layer(merged_output)
    weight_layer_other = create_weighted_layer(4)
    final_output_other = weight_layer_other(merged_output_other)

    # 返回自身
    current_self_layer = Lambda(lambda tensors: tensors, name='current_self_layer')
    final_output = current_self_layer(final_output)
    other_self_layer = Lambda(lambda tensors: tensors, name='other_self_layer')
    final_output_other = other_self_layer(final_output_other)

    #  模型构建
    ## 输入输出
    model = Model(inputs=[continuous_baseline_data, continuous_baseline_day_data, continuous_baseline_week_data, wind_data, weather_data, day_data, hour_data, havePeople_data, continuous_baseline_data_other, continuous_baseline_day_data_other, continuous_baseline_week_data_other, wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other], outputs=[final_output, final_output_other])
    ## compile
    opt = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    #model.compile(optimizer=opt, loss='mae', metrics=['mae', 'mse', 'mape'])  # adam
    model.compile(optimizer=opt,
                  loss={
                      'current_self_layer': 'mae',
                      'other_self_layer': 'mae'
                  },
                  loss_weights={
                      'current_self_layer': 1.,
                      'other_self_layer': 1.
                  },
                  metrics=['mae', 'mse', 'mape'])
    
    return model


def model_train_MDL_DNN(save_folder, multi_time_train_label, multi_time_train_t_0_data, baseline_std_cols, windows_len, MDL_baseline_data, multi_time_train_t_0_label_other, multi_time_train_t_0_data_other, MDL_baseline_data_other):

    input_features_size = len(baseline_std_cols)

    label_next_data, label_next_std_data, label_current_data = multi_time_train_label

    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_train_label
    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_train_t_0_data

    #print('multi_time_label_next_data\n', multi_time_label_next_data)


    # MDL data
    continuous_baseline_day_data, continuous_baseline_week_data = MDL_baseline_data


    # other stations
    label_next_data_other, label_next_std_data_other, label_current_data_other = multi_time_train_t_0_label_other

    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_train_label
    continuous_model_data_other, continuous_baseline_data_other, wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other = multi_time_train_t_0_data_other

    continuous_baseline_day_data_other, continuous_baseline_week_data_other = MDL_baseline_data_other



    # model_build
    model = model_build_MDL(input_features_size, windows_len)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_folder + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # model.fit
    model.fit([continuous_baseline_data, continuous_baseline_day_data, continuous_baseline_week_data, wind_data, weather_data, day_data, hour_data, havePeople_data, continuous_baseline_data_other, continuous_baseline_day_data_other, continuous_baseline_week_data_other, wind_data_other, weather_data_other, day_data_other, hour_data_other, havePeople_data_other],
              [label_next_std_data, label_next_std_data_other],
              epochs=300,  # 避免过拟合，减小循环次数
              batch_size=64,
              #callbacks=[save_best, early_stop],
              callbacks=[early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
