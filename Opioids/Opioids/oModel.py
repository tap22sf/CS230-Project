from keras import layers
from keras.layers import Input, Embedding, LSTM, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers


def OpioidModel(input_shape, layers, nodes, dropout_rate, l2reg):
    """
    Implementation of the OpioidModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = Dropout(dropout_rate)(X_input)
    X = Dense(nodes, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l2(l2reg))(X)

    # Variable number of layers used during arch sensitivity testing
    for i in range(layers):
        X = Dropout(dropout_rate)(X)
        X = Dense(2*nodes, activation='relu', kernel_regularizer=regularizers.l2(l2reg))(X)
        
    predictions = Dense(1, activation='sigmoid')(X)

    # Create model
    model = Model(inputs=X_input, outputs=predictions, name='OpioidModel')
    
    return model
