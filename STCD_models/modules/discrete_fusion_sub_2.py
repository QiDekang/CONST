from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level, l2_rate


#  孪生网络融合网络
def create_discrete_fusion_layers_sub_2(fusion_feature_size, windows_len):

    #embedded_size = 32
    size_one = 32
    size_two = 32
    size_three = 32
    size_four = 32

    #  input
    
    input_features = Input(shape=(windows_len, fusion_feature_size),
                           dtype='float')
    #  layers
    hiddenLayer_all = Dense(size_three, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(input_features)
    hiddenLayer_all = Dense(size_four, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_all)
    hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    one_time_output = Dense(1, kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_all)
    ## (batch_size, windows_len, 1) -- > (batch_size, windows_len)
    one_time_output = Reshape(target_shape=(windows_len,))(one_time_output)
    multi_time_output = Dense(1, kernel_regularizer=regularizers.l2(l2_rate))(one_time_output)

    model = Model(inputs=input_features, outputs=multi_time_output)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
