#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
# 

import numpy as np
import tensorflow as tf

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import *

import keras.backend as K

from dataParser import *
from oModel import *

import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###################################################################################################
# Parameters

# General
training_portion = 0.9
seed = 0 # Random generator

# Optimization parameters
epochs=3
batch_size=100

# Adam parameters 
lr = 0.05
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)

###################################################################################################

# Read from folder "Medical Database" which is at the same level as the project folder
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(fileDir)))
dataFile1Path = os.path.join(parentDir, 'Medical Database/scaled')
dataFile2Path = os.path.join(parentDir, 'Medical Database/data_y')

# Read the datasets
X,Y = loadData(dataFile1Path, dataFile2Path)
print ("X shape: " + str(X.shape))
print ("Y shape: " + str(Y.shape))

# Normalize input and output fields


# Divide into train/dev/test sets
X_training, Y_training, X_validation, Y_validation = split_dataset(X, Y, training_portion, seed)
print ("X_training shape: " + str(X_training.shape))
print ("Y_training shape: " + str(Y_training.shape))
print ("X_validation shape: " + str(X_validation.shape))
print ("Y_validation shape: " + str(Y_validation.shape))

# Build a model 
oModel = OpioidModel(X[0].shape)

# Apply input
# Optimizer values

optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

oModel.compile(optimizer=optimizer, loss = "binary_crossentropy", metrics = ["binary_accuracy"])

# Train the model, iterating on the data in batches of 32 samples
history = oModel.fit(X_training, Y_training, epochs=epochs, batch_size=batch_size)

# Evaluate the results
preds = oModel.evaluate(X_validation, Y_validation)
oModel.summary()

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print()
#for layer in oModel.layers:
#    weights = layer.get_weights() # list of numpy arrays
#    print(weights)

# History
# list all data in history
# print(history.history.keys())
print()

# summarize history for accuracy
# plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()



