from common.config import dropout_rate, verbose_level
from keras import regularizers
from keras import optimizers, callbacks, initializers
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras import layers


#  离散特征嵌入
def create_embedding_layers(discrete_feature_num, embed_size):

    #  input
    input_wind = Input(shape=(1,), dtype='int32', name='input_wind')
    input_weather = Input(shape=(1,), dtype='int32', name='input_weather')
    input_day = Input(shape=(1,), dtype='int32', name='input_day')
    input_hour = Input(shape=(1,), dtype='int32', name='input_hour')
    input_people = Input(shape=(1,), dtype='int32', name='input_people')

    #  使用离散特征
    embedded_wind = Embedding(
        input_dim=9, output_dim=embed_size)(input_wind)
    embedded_weather = Embedding(
        input_dim=6, output_dim=embed_size)(input_weather)
    embedded_day = Embedding(
        input_dim=7, output_dim=embed_size)(input_day)
    embedded_hour = Embedding(
        input_dim=24, output_dim=embed_size)(input_hour)
    embedded_people = Embedding(
        input_dim=2, output_dim=embed_size)(input_people)
    embedded_wind = Reshape(target_shape=(embed_size,))(embedded_wind)
    embedded_weather = Reshape(target_shape=(embed_size,))(embedded_weather)
    embedded_day = Reshape(target_shape=(embed_size,))(embedded_day)
    embedded_hour = Reshape(target_shape=(embed_size,))(embedded_hour)
    embedded_people = Reshape(target_shape=(embed_size,))(embedded_people)
    merged_features = layers.concatenate(
        [embedded_wind, embedded_weather, embedded_day, embedded_hour, embedded_people], axis=1)  # 从第1个维度拼接

    model = Model(inputs=[input_wind, input_weather, input_day,
                  input_hour, input_people], outputs=merged_features)
    embedded_features_size = discrete_feature_num * embed_size

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model, embedded_features_size
