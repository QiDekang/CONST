from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level


#  孪生网络融合网络
def create_weighted_layer(merged_size):

    #  input
    input_features = Input(shape=(merged_size,),
                           dtype='float', name='input_features')
    #  network
    #### 网络最后一层不要用relu，使用默认的linear
    output = Dense(1, kernel_regularizer=regularizers.l2(0.01))(input_features)
    
    # model
    model = Model(inputs=input_features, outputs=output)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
