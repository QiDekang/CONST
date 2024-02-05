from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, LSTM
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level, l2_rate


#  孪生网络融合网络
def create_LSTM_fusion_layers_deep(input_features_size, windows_len):

    #  input
    LSTM_input_features = Input(shape=(windows_len, input_features_size), dtype='float')

    #  layers
    hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate))(LSTM_input_features)
    hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer)
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), name='LSTM_fusion_layers_1')(LSTM_input_features)
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), name='LSTM_2')(hiddenLayer)
    hiddenLayer = Dropout(rate=dropout_rate)(hiddenLayer)
    LSTM_output = Dense(1, kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer)

    model = Model(inputs=LSTM_input_features, outputs=LSTM_output)

    #  单独使用必须有compile，作为基础模型需要去掉compile

    return model
