#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
# 

import numpy as np
import tensorflow as tf

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import *
from keras.models import load_model

import keras.backend as K

from dataParser import *
from oModel import *

import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

###################################################################################################
# Parameters

# General
train = True
weightfile = 'omodel_weights.h5'
training_portion = 0.9
seed = 0 # Random generator

# Optimization parameters

epochs= 100
batch_size=1000
loss = "binary_crossentropy"
#loss = "mean_squared_error"

# Adam parameters 
lr = 0.002
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)
optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipvalue=1)

#lr = 0.002
#optimizer = sgd(lr=lr)


##########################################################################################&#########

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
print ("Y Train mean: " + str(Y.mean()))

# Divide into train/dev/test sets
X_training, Y_training, X_validation, Y_validation = split_dataset(X, Y, training_portion, seed)
print ("X_training shape: " + str(X_training.shape))
print ("Y_training shape: " + str(Y_training.shape))
print ("X_validation shape: " + str(X_validation.shape))
print ("Y_validation shape: " + str(Y_validation.shape))

np.savetxt ("Test_y.csv", Y_validation)

# Train the model, iterating on the data in batches of 32 samples
if train:
    # Build a model 
    oModel = OpioidModel(X[0].shape)

    # Optimizer values
    oModel.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', mean_pred])

    ## Add a callback for saving weights each epoch
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{binary_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit.
    history = oModel.fit(X_training, Y_training, epochs=epochs, shuffle=True, batch_size=batch_size, callbacks=callbacks_list, validation_split = 0.1)


    oModel.save(weightfile)
else:
    oModel = load_model(weightfile, custom_objects={'mean_pred': mean_pred})

# Evaluate the results
preds = oModel.evaluate(X_validation, Y_validation)
oModel.summary()

print()
print ("Test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#print()

preds = oModel.evaluate(X_training, Y_training)
print("Preds str = " + str(preds))
print ("Training Loss = " + str(preds[0]))
print ("Trainign Accuracy = " + str(preds[1]))

#print()
#for layer in oModel.layers:
#    weights = layer.get_weights() # list of numpy arrays
#    print(weights)

# History
# list all data in history
# print(history.history.keys())
Y_pred = oModel.predict(X_validation, batch_size=None, verbose=0, steps=None) 
Y_pred_binary = (Y_pred > 0.5)

print("Y Predictions")
print ("Y Pred Mean : " + str(Y_pred.mean()))
np.savetxt ("Pred_y.csv", Y_pred)

print("Y Predictions - binary")
print ("Y Pred Binary Mean : " + str(Y_pred_binary.mean()))
np.savetxt ("Pred_binary_y.csv", Y_pred_binary)

Y_pred_train = oModel.predict(X_training, batch_size=None, verbose=0, steps=None) 
print ("Y Pred train Mean : " + str(Y_pred_train.mean()))
np.savetxt ("Pred_train_y.csv", Y_pred_train)

# summarize history for accuracy
#plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
print(history.history.keys())


# # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_binary_accuracy'])
plt.plot(history.history['binary_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'dev loss', 'dev accuracy', 'training accuracy'], loc='upper left')
plt.show()

