from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    
    ################################################################################################
    # TODO: Add batch normalization
        #sources: https://keras.io/layers/normalization/ and pattern used in cnn_rnn_model below
        # Add batch normalization: bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
        
    bn_rnn = BatchNormalization(name = 'bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
        #layer: bn_rnn
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    ################################################################################################   
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    ################################################################################################
    #Same code used above in rnn_model
    # TODO: Add batch normalization    
    bn_rnn = BatchNormalization(name = 'bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    ################################################################################################   
    
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    ################################################################################################
    # TODO: Add recurrent layers, each with batch normalization
    #...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    #time_dense = ...
    ################################################################################################    
    
    ################################################################################################
    #From vui_notebook: As a quick check that you have implemented the additional functionality in
    #deep_rnn_model correctly, make sure that the architecture that you specify here is identical
    #to rnn_model if recur_layers=1
    
    #Therefore we need use the lines used in rnn_model and just add the additional layers
    
    # Add recurrent layer
    simp_rnn = GRU(units, return_sequences=True, implementation=2, name='rnn')(input_data)
                   #activation=activation, we take off this parameter since it no longer appears as input in the function
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name = 'bn_deep_rnn')(simp_rnn) #Changed name form 'bn_simp_rnn' to 'bn_deep_rnn'
    
    ################
    #New part
    if recur_layers == 0:
        print("Must have at least 1 recurrent layer!")
    #If recur_layers = 1, we already have all we nedd
    if recur_layers > 1:
        #range(recur_layers) = [0, recur_layers), range(recur_layers - 1) = [0, recur_layers - 1)
        #We already have 1, we must have 2, 3, 4, ... recur_layers.
        for i in range(2, recur_layers + 1):
            simp_rnn = GRU(units, return_sequences=True, implementation=2, name='rnn' + str(i))(bn_rnn)
            bn_rnn = BatchNormalization(name = 'bn_deep_rnn' + str(i))(simp_rnn)        
    #End of new part
    ################
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    ################################################################################################   
       
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    #################################################################################################
    # TODO: Add bidirectional recurrent layer
    #bidir_rnn = ...
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    #time_dense = ...
    #################################################################################################

    #################################################################################################
    # TODO: Add bidirectional recurrent layer
    #https://keras.io/layers/wrappers/
    #From vui_notebook: Feel free to use SimpleRNN, LSTM, or GRU units.
                        #When specifying the Bidirectional wrapper, use merge_mode='concat'.
    #Using codes from model_final from Project 3 (Bidirectional) and from rnn_model above
    bidir_rnn = bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, implementation=2, name='bidir_rnn'))(input_data)
    bn_bidir_rnn = BatchNormalization(name='bn_bidir')(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_bidir_rnn)
    #################################################################################################
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers, dropout_rate, output_dim=29):

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)


    # Add recurrent layer
    simp_rnn = LSTM(units, return_sequences=True, implementation=2, name='rnn', dropout=dropout_rate)(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name = 'bn_deep_rnn')(simp_rnn) 
   
    ################
    #New part
    if recur_layers == 0:
        print("Must have at least 1 recurrent layer!")
    #If recur_layers = 1, we already have all we nedd
    if recur_layers > 1:
        #range(recur_layers) = [0, recur_layers), range(recur_layers - 1) = [0, recur_layers - 1)
        #We already have 1, we must have 2, 3, 4, ... recur_layers.
        for i in range(2, recur_layers + 1):
            simp_rnn = LSTM(units, return_sequences=True, implementation=2, name='rnn' + str(i), dropout=dropout_rate)(bn_rnn)
            bn_rnn = BatchNormalization(name = 'bn_deep_rnn' + str(i))(simp_rnn)        
    #End of new part
    ################
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)   
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model








