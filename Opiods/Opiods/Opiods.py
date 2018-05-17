#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
# 

import numpy as np

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K

from dataParser import *
from oModel import *



# Read the datasets
X,Y = loadData(r'E:\Medical Database\df_x_withGeo', r'E:\Medical Database\data_y')

# Normalize input and output fields
print ("number of training examples = " + str(X.shape[0]))
print ("X shape: " + str(X.shape))
print (X)
print (X.dtype)

print ("Y shape: " + str(Y.shape))
print (Y)

# Divide into train/dev/test sets

# Build a model
oModel = OpiodModel(X[0].shape)

# Apply input
oModel.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["binary_accuracy"])

# Train the model, iterating on the data in batches of 32 samples
oModel.fit(X, Y, epochs=4, batch_size=1000)

# Evaluate the results
#preds = oModel.evaluate(X_test, Y_test)

#print()
#print ("Loss = " + str(preds[0]))
#print ("Test Accuracy = " + str(preds[1]))

#oModel.summary()
