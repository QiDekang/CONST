from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level


#  孪生网络融合网络
def create_fusion_network(embedded_size):

    #embedded_size = 32
    size_one = 32
    size_two = 32
    size_three = 32

    #  input
    input_features = Input(shape=(embedded_size,),
                           dtype='float', name='input_features')
    #  network
    hiddenLayer_all = Dense(size_one, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_all = Dense(size_two, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hiddenLayer_all)
    hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_all = Dense(size_three, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hiddenLayer_all)
    hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    output = Dense(1, name='siamese_output', kernel_regularizer=regularizers.l2(0.01))(hiddenLayer_all)
    model = Model(inputs=input_features, outputs=output)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
