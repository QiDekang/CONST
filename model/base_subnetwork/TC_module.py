from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Subtract
from keras.models import Model
from keras import callbacks, initializers
from keras import regularizers
from tensorflow import optimizers
from common.config import dropout_rate, verbose_level
from model.base_subnetwork.embedding_layers import create_embedding_layers
from model.base_subnetwork.fusion_network import create_fusion_network


###################################################
# 时间一致性网络作为基础模块，用于多时刻融合网络中。
###################################################
def create_TC_module(input_features_size):

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

    input_long = Input(shape=(input_features_size,), dtype='float', name='input_long')
    input_long_wind = Input(shape=(1,), dtype='int32', name='input_long_wind')
    input_long_weather = Input(shape=(1,), dtype='int32', name='input_long_weather')
    input_long_day = Input(shape=(1,), dtype='int32', name='input_long_day')
    input_long_hour = Input(shape=(1,), dtype='int32', name='input_long_hour')
    input_long_people = Input(shape=(1,), dtype='int32', name='input_long_people')

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
    embedding_long = embedding_layers(
        [input_long_wind, input_long_weather, input_long_day, input_long_hour, input_long_people])
    ## 连续特征已经提前做差，现对离散特征embed后做差
    short_discrete_feature = Subtract(name='subtract_short_layer')([embedding_current, embedding_short])
    long_discrete_feature = Subtract(name='subtract_long_layer')([embedding_current, embedding_long])
    ## 拼接连续特征和离散特征
    merged_short_features = layers.concatenate([input_short, short_discrete_feature], axis=1)  # 从第1个维度拼接
    merged_long_features = layers.concatenate([input_long, long_discrete_feature], axis=1)  # 从第1个维度拼接

    # 多任务学习，共享参数
    ## 创建网络
    fusion_feature_size = input_features_size + embedded_features_size
    fusion_network = create_fusion_network(fusion_feature_size)
    fusion_short_prediction = fusion_network(merged_short_features)
    fusion_long_prediction = fusion_network(merged_long_features)

    # 输入输出
    model = Model(inputs=[input_current_wind, input_current_weather, input_current_day, input_current_hour, input_current_people, input_short, input_short_wind, input_short_weather, input_short_day,
                  input_short_hour, input_short_people, input_long, input_long_wind, input_long_weather, input_long_day, input_long_hour, input_long_people], outputs=[fusion_short_prediction, fusion_long_prediction])

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
