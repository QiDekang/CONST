from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level


#  孪生网络融合网络
def create_feature_extraction_layer(input_size, feature_extraction_layer_size):

    #feature_extraction_layer_size = 32

    #  input
    input_features = Input(shape=(input_size,),
                           dtype='float', name='input_features')
    #  network
    output = Dense(feature_extraction_layer_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    
    model = Model(inputs=input_features, outputs=output)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
