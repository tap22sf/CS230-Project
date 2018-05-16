#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
# 

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# Read the datasets
X,Y = loadData()

# Normalize input and output fields
print ("number of training examples = " + str(X.shape[0]))
print ("X shape: " + str(X.shape))
print (X)
print ("Y shape: " + str(Y.shape))
print (Y)

# Normalize input and output datasets

# Divide into train/dev/test sets

# Build a model
#oModel = OpiodModel(X_train[0].shape)

# Apply input
#oModel.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["binary_accuracy"])

# Train the model, iterating on the data in batches of 32 samples
#oModel.fit(X_train, Y_train, epochs=40, batch_size=16)

# Evaluate the results
#preds = oModel.evaluate(X_test, Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

oModel.summary()
