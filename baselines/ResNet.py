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
from STCD_models.modules.MDL_DNN_fusion_layers_sub_1 import create_MDL_fusion_layers_sub_1
from STCD_models.modules.MDL_DNN_fusion_layers_sub_2 import create_MDL_fusion_layers_sub_2
from STCD_models.modules.MDL_fusion_layers import create_MDL_fusion_layers


################
# DNN with indoor temperature
# 输入为当前时刻的连续特征（实际值）以及离散特征。输出为t+1时刻的室温。
################
def model_build_ResNet(input_features_size, windows_len):

    # input
    input_current = Input(shape=(windows_len, input_features_size), dtype='float', name='input_current') # 当前时刻连续特征的实际值
    input_current_wind = Input(shape=(windows_len,), dtype='int32', name='input_current_wind')
    input_current_weather = Input(shape=(windows_len,), dtype='int32', name='input_current_weather')
    input_current_day = Input(shape=(windows_len,), dtype='int32', name='input_current_day')
    input_current_hour = Input(shape=(windows_len,), dtype='int32', name='input_current_hour')
    input_current_people = Input(shape=(windows_len,), dtype='int32', name='input_current_people')

    # 预处理。使用离散特征嵌入层
    ## embedding_layers
    #discrete_feature_num = 5  #离散特征数量
    #embed_size = 2  #离散特征嵌入后输出的维度。
    embedding_layers, embedded_features_size = create_embedding_layers(windows_len)
    ##  embedding
    embedding_current = embedding_layers(
        [input_current_wind, input_current_weather, input_current_day, input_current_hour, input_current_people])
    #print(embedding_current)
    #print(input_current)
    ## 拼接连续特征和离散特征
    merged_current_features = layers.concatenate([input_current, embedding_current], axis=2)  # 从第2个维度拼接

    # 融合网络
    ## 创建网络
    fusion_feature_size = input_features_size + embedded_features_size
    #fusion_network = create_fusion_layers(fusion_feature_size, windows_len)
    #fusion_network = create_fusion_layers_deep(fusion_feature_size, windows_len)
    fusion_network = create_MDL_fusion_layers(fusion_feature_size, windows_len)
    fusion_current_prediction = fusion_network(merged_current_features)


    #  模型构建
    ## 输入输出
    model = Model(inputs=[input_current, input_current_wind, input_current_weather, input_current_day, input_current_hour, input_current_people], outputs=[fusion_current_prediction])
    ## compile
    opt = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mae', metrics=['mae', 'mse', 'mape'])  # adam
    
    return model


def model_train_ResNet(save_folder, multi_time_train_label, multi_time_train_t_0_data, baseline_std_cols, windows_len):

    input_features_size = len(baseline_std_cols)

    label_next_data, label_next_std_data, label_current_data = multi_time_train_label

    #multi_time_label_next_data, multi_time_label_next_std_data, multi_time_label_current_data, multi_time_label_diff_data, multi_time_label_diff_std_data = multi_time_train_label
    continuous_model_data, continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data = multi_time_train_t_0_data

    #print('multi_time_label_next_data\n', multi_time_label_next_data)

    # model_build
    model = model_build_ResNet(input_features_size, windows_len)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_folder + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # model.fit
    model.fit([continuous_baseline_data, wind_data, weather_data, day_data, hour_data, havePeople_data],
              label_next_std_data,
              epochs=300,  # 避免过拟合，减小循环次数
              batch_size=64,
              callbacks=[save_best, early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
