from common.config import dropout_rate, verbose_level, embed_size
from keras import regularizers
from keras import optimizers, callbacks, initializers
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Concatenate
from keras import layers


#  离散特征嵌入
def create_embedding_layers(windows_len):

    #  input
    input_wind = Input(shape=(windows_len,), dtype='int32')
    input_weather = Input(shape=(windows_len,), dtype='int32')
    input_day = Input(shape=(windows_len,), dtype='int32')
    input_hour = Input(shape=(windows_len,), dtype='int32')
    input_people = Input(shape=(windows_len,), dtype='int32')

    #  使用离散特征
    embedded_wind = Embedding(
        input_dim=9, output_dim=embed_size, input_length=windows_len)(input_wind)
    embedded_weather = Embedding(
        input_dim=6, output_dim=embed_size, input_length=windows_len)(input_weather)
    embedded_day = Embedding(
        input_dim=7, output_dim=embed_size, input_length=windows_len)(input_day)
    embedded_hour = Embedding(
        input_dim=24, output_dim=embed_size, input_length=windows_len)(input_hour)
    # 不使用特征：室内是否有人
    #embedded_people = Embedding(
    #    input_dim=2, output_dim=embed_size, input_length=windows_len)(input_people)
    '''
    embedded_wind = Reshape(target_shape=(embed_size,))(embedded_wind)
    embedded_weather = Reshape(target_shape=(embed_size,))(embedded_weather)
    embedded_day = Reshape(target_shape=(embed_size,))(embedded_day)
    embedded_hour = Reshape(target_shape=(embed_size,))(embedded_hour)
    embedded_people = Reshape(target_shape=(embed_size,))(embedded_people)
    '''
    #merged_features = layers.concatenate([embedded_wind, embedded_weather, embedded_day, embedded_hour, embedded_people], axis=2)  # 从第2个维度拼接，batch_size, input_length, output_dim
    #merged_features = layers.concatenate([embedded_wind, embedded_weather, embedded_day, embedded_hour], axis=2, name='embedded_merged_features')  # 从第2个维度拼接，batch_size, input_length, output_dim
    merged_features = Concatenate(axis=2)([embedded_wind, embedded_weather, embedded_day, embedded_hour])  # 从第2个维度拼接，batch_size, input_length, output_dim

    print(merged_features)

    model = Model(inputs=[input_wind, input_weather, input_day,
                  input_hour, input_people], outputs=merged_features)
    #embedded_features_size = 5 * embed_size
    embedded_features_size = 4 * embed_size

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model, embedded_features_size
