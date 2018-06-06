from keras import layers
from keras.layers import Input, Embedding, LSTM, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def OpioidModel(input_shape, layers, nodes, dropout_rate):
    """
    Implementation of the OpioidModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = Dense(nodes, input_shape=input_shape, activation='relu')(X_input)
    #X = BatchNormalization()(X)

    # Variable number of layers used during arch sensitivity testing
    for i in range(layers):
        X = Dropout(dropout_rate)(X)
        X = Dense(2*nodes, activation='relu')(X)
        
    predictions = Dense(1, activation='sigmoid')(X)

    # Create model
    model = Model(inputs = X_input, outputs = predictions, name='OpioidModel')
    
    return model
