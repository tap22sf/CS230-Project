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

    X = Dense(500, input_shape= input_shape, activation='relu', name='fc1')(X_input)
    X = Dense(100, activation='relu', name='fc2')(X)
    predictions = Dense(1, activation='softmax')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = predictions, name='OpiodModel')
    
    return model
