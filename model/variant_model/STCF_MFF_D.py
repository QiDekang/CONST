from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Subtract
from keras.models import Model
from keras import callbacks, initializers
from keras import regularizers
from tensorflow import optimizers
from common.config import dropout_rate, verbose_level
from model.base_subnetwork.embedding_layers import create_embedding_layers
from model.base_subnetwork.fusion_network import create_fusion_network


def model_build_STCF_MFF_D(input_features_size):

    # input
    input_current_wind = Input(shape=(1,), dtype='int32', name='input_current_wind')
    input_current_weather = Input(shape=(1,), dtype='int32', name='input_current_weather')
    input_current_day = Input(shape=(1,), dtype='int32', name='input_current_day')
    input_current_hour = Input(shape=(1,), dtype='int32', name='input_current_hour')
    input_current_people = Input(shape=(1,), dtype='int32', name='input_current_people')

    input_short = Input(shape=(input_features_size,), dtype='float', name='input_short')
    input_short_wind = Input(shape=(1,), dtype='int32', name='input_short_wind')
    input_short_weather = Input(shape=(1,), dtype='int32', name='input_short_weather')
    input_short_day = Input(shape=(1,), dtype='int32', name='input_short_day')
    input_short_hour = Input(shape=(1,), dtype='int32', name='input_short_hour')
    input_short_people = Input(shape=(1,), dtype='int32', name='inpu_short_people')

    # 预处理。使用共享参数的离散特征嵌入
    ## embedding_layers, 特征嵌入
    discrete_feature_num = 5
    embed_size = 2
    embedding_layers, embedded_features_size = create_embedding_layers(discrete_feature_num, embed_size)
    ##  embedding
    embedding_current = embedding_layers(
        [input_current_wind, input_current_weather, input_current_day, input_current_hour, input_current_people])
    embedding_short = embedding_layers(
        [input_short_wind, input_short_weather, input_short_day, input_short_hour, input_short_people])
    ## 连续特征已经提前做差，现对离散特征embed后做差
    short_discrete_feature = Subtract(name='subtract_short_layer')([embedding_current, embedding_short])
    ## 拼接连续特征和离散特征
    merged_short_features = layers.concatenate([input_short, short_discrete_feature], axis=1)  # 从第1个维度拼接

    # 差分网络
    ## 创建网络
    fusion_feature_size = input_features_size + embedded_features_size
    fusion_network = create_fusion_network(fusion_feature_size)
    fusion_short_prediction = fusion_network(merged_short_features)


    ####  模型构建
    # 输入输出
    model = Model(inputs=[input_current_wind, input_current_weather, input_current_day, input_current_hour, input_current_people, input_short, input_short_wind, input_short_weather, input_short_day,
                  input_short_hour, input_short_people], outputs=[fusion_short_prediction])
    # compile
    opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                          decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mae', metrics=['mae', 'mse', 'mape'])  # adam
    
    return model


def model_train_STCF_MFF_D(save_path, train_current, train_short, train_long, feature_cols, label_diff):

    input_features_size = len(feature_cols)

    # model_build
    model = model_build_STCF_MFF_D(input_features_size)

    # early_stop
    save_best = callbacks.ModelCheckpoint(
        save_path + 'bestModel.h5', monitor='val_loss', verbose=verbose_level, save_best_only=True, mode='min')
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=verbose_level)

    # model.fit
    model.fit([train_current["wind_direction"], train_current["weather"], train_current["day"], train_current["hour"], train_current["havePeople"], train_short[feature_cols], train_short["wind_direction"], train_short["weather"], train_short["day"], train_short["hour"], train_short["havePeople"]],
              [train_short[label_diff]],
              epochs=300,  # 避免过拟合，减小循环次数
              batch_size=32,
              #callbacks=[save_best, early_stop],
              callbacks=[early_stop],
              validation_split=0.2,
              verbose=verbose_level)

    return model
