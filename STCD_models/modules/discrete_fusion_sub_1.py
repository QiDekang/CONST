from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level, l2_rate


#  孪生网络融合网络
def create_discrete_fusion_layers_sub_1(fusion_feature_size, windows_len):

    #embedded_size = 32
    size_one = 32
    size_two = 32
    size_three = 32

    #  input
    
    input_features = Input(shape=(windows_len, fusion_feature_size),
                           dtype='float')
    #  layers
    ## (batch_size, ..., input_dim) --> (batch_size, ..., units)
    #hiddenLayer_all = Dense(size_one, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    hiddenLayer_all = Dense(size_one, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(input_features)
    hiddenLayer_all = Dense(size_two, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_all)
    #hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    #hiddenLayer_all = Dense(size_three, activation='relu', kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_all)

    model = Model(inputs=input_features, outputs=hiddenLayer_all)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
