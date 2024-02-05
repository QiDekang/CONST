from keras import layers
from keras.layers import Input, Dense, Lambda, Dropout, Embedding, Reshape, Add, LSTM
from keras.models import Model
from keras import optimizers, callbacks, initializers
from keras import regularizers
from common.config import dropout_rate, verbose_level, l2_rate


#  孪生网络融合网络
def create_MDL_fusion_layers_sub_1(input_features_size, windows_len):

    #embedded_size = 32
    size_one = 32
    size_two = 32
    size_three = 32
    size_four = 32
    size_five = 32
    size_six = 32

    #  input
    
    input_features = Input(shape=(windows_len, input_features_size),
                           dtype='float')
    #  layers
    '''
    ## (batch_size, ..., input_dim) --> (batch_size, ..., units)
    #hiddenLayer_all = Dense(size_one, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_features)
    hiddenLayer_all_1 = Dense(size_one, activation='relu')(input_features)
    #hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_all_2 = Dense(size_two, activation='relu')(hiddenLayer_all_1)
    #hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_all_3 = Dense(size_three, activation='relu')(hiddenLayer_all_2)
    #hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_add_1 = Add()([hiddenLayer_all_3, hiddenLayer_all_1])
    hiddenLayer_all_4 = Dense(size_four, activation='relu')(hiddenLayer_add_1)
    hiddenLayer_all_5 = Dense(size_five, activation='relu')(hiddenLayer_all_4)
    #hiddenLayer_all = Dropout(rate=dropout_rate)(hiddenLayer_all)
    hiddenLayer_all_6 = Dense(size_six, activation='relu')(hiddenLayer_all_5)
    hiddenLayer_all_7 = Dense(size_six, activation='relu')(hiddenLayer_all_6)
    hiddenLayer_add_2 = Add()([hiddenLayer_all_7, hiddenLayer_all_5])
    hiddenLayer_all_8 = Dense(size_six, activation='relu')(hiddenLayer_add_2)
    

    model = Model(inputs=input_features, outputs=hiddenLayer_all_8)
    '''
    hiddenLayer_1 = LSTM(32, input_shape=(windows_len, input_features_size), return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate))(input_features)
    hiddenLayer_2 = LSTM(input_features_size, input_shape=(windows_len, size_one), return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_1)
    hiddenLayer_add = Add()([hiddenLayer_2, input_features])
    #hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), name='LSTM_fusion_layers_1')(LSTM_input_features)
    hiddenLayer = LSTM(32, input_shape=(windows_len, input_features_size), return_sequences=True, kernel_regularizer=regularizers.l2(l2_rate))(hiddenLayer_add)

    #  单独使用必须有compile，作为基础模型需要去掉compile
    model = Model(inputs=input_features, outputs=hiddenLayer)

    return model
