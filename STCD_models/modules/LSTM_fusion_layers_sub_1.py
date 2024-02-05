from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, LSTM
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level, l2_rate


#  孪生网络融合网络
def create_LSTM_fusion_layers_sub_1(input_features_size, windows_len):

    #  input
    #LSTM_input_features = Input(shape=(windows_len, input_features_size), dtype='float', name='input_features_sub_1')
    LSTM_input_features = Input(shape=(windows_len, input_features_size), dtype='float')

    #  layers
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), return_sequences=True, name='LSTM_fusion_layers_sub_1', kernel_regularizer=regularizers.l2(l2_rate))(LSTM_input_features)
    hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate))(LSTM_input_features)
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), name='LSTM_fusion_layers_1')(LSTM_input_features)
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), name='LSTM_2', kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer)
    #LSTM_output = Dense(1, name='LSTM_fusion_layers_output')(hiddenLayer)

    model = Model(inputs=LSTM_input_features, outputs=hiddenLayer)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
