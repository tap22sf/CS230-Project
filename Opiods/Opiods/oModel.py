from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

def OpiodModel(input_shape):
    """
    Implementation of the OpiodModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = Dense(1000, input_shape= input_shape, activation='sigmoid', name='fc1')(X_input)
    #X = Dense(5000, activation='sigmoid', name='fc4')(X)
    #X = Dense(1000, activation='sigmoid', name='fc5')(X)
    #X = Dense(1000, activation='sigmoid', name='fc6')(X)
    #X = Dense(10, activation='sigmoid', name='fc7')(X)
    predictions = Dense(1, activation='sigmoid')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = predictions, name='OpiodModel')
    
    return model
