from keras import layers
from keras.layers import Input, Embedding, LSTM, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def OpioidModel(input_shape):
    """
    Implementation of the OpioidModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

    X = Dense(2048, input_shape=input_shape, activation='relu', name='fc1')(X_input)
    X = Dropout(.3)(X)
    #X = BatchNormalization()(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(.3)(X)
    X = Dense(512, activation='relu')(X)
    #X = Dense(512, activation='relu')(X)
    #X = Dense(512, activation='relu')(X)
    #X = Dense(512, activation='relu')(X)
    #X = Dense(1000, activation='sigmoid', name='fc5')(X)
    #^X = Dense(1000, activation='sigmoid', name='fc6')(X)
    predictions = Dense(1, activation='sigmoid')(X)
    #predictions = Dense(1, input_shape= input_shape, activation='sigmoid')(X_input)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = predictions, name='OpioidModel')
    
    return model
