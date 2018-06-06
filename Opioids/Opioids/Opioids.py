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


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

###################################################################################################
# Parameters

# General
train = True
training_portion = .98
test_portion = 0.01

# Amount of training set to holdout for validation(dev)
validation_split = 0.1

weightDir = 'Weights'
resultsDir = 'Results'

seed = 0 # Random generator

# Prameters
# Perferred defaults
bz = 1024
epochs = 40

lr = -3.0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)
loss = "binary_crossentropy"

# Model
layers = 2
nodes = 2000
dropout = 0

# Hyperparameters - list of dictionaries to setup training runs
parameters = []


# batch size scane
#for bz in (128, 256, 512, 1024, 2048):
#    run = {'epochs':10,'batch':bz,'lr' :lr,'layers':2,'nodes':2000, 'dropout':dropout}
#    parameters.append (run)

## Learning rate senstivity tests
#for lr in (-2, -3, -4, -5, -6):
#    run = {'epochs':10,'batch':1024,'lr' :lr,'layers':2,'nodes':2000, 'dropout':dropout}
#    parameters.append (run)

# Node size and layer sensitivity testing
for n in (500, 1000, 2000, 3000):
    for l in (0, 1, 2):
        lr = -3.0
        epochs = 100
        dropout = 0.5
        run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':l,'nodes':n, 'dropout':dropout}
        parameters.append (run)




##########################################################################################&#########

# Read from folder "Medical Database" which is at the same level as the project folder
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(fileDir)))
dataFile1Path = os.path.join(parentDir, 'Medical Database\\withEncodings')
dataFile2Path = os.path.join(parentDir, 'Medical Database\\data_y')

# Read the datasets
X, Y = loadData(xFilename=dataFile1Path, yFilename=dataFile2Path)
print("X shape: " + str(X.shape))
print("Y shape: " + str(Y.shape))
print("Y Train mean: " + str(Y.mean()))

# Divide into train/dev/test sets
X_test, Y_test, X_trainDev, Y_trainDev = split_dataset(X, Y, split1=test_portion, split2=training_portion, seed=seed)
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
print("X_trainDev shape: " + str(X_trainDev.shape))
print("Y_trainDev shape: " + str(Y_trainDev.shape))

X_train = X_trainDev[0: int((1-validation_split) * X_trainDev.shape[0]), :]
Y_train = Y_trainDev[0: int((1-validation_split) * X_trainDev.shape[0]), :]
X_dev = X_trainDev[int((1 - validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]
Y_dev = Y_trainDev[int((1 - validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]

#np.savetxt (resultsDir + r'\Test_y.csv', Y_test)

# Train the model, iterating on the data in batches based on the run database
if train:
    for run in parameters:
        epochs = run ['epochs']
        batch = run ['batch']
        lr = run['lr']
        layers = run['layers']
        nodes = run['nodes']
        dr = run['dropout']

        # adjusting run variables                    
        lr = np.power(float(10), np.array(lr))
        print("=== New run" +
                "  lr="         + str(lr) +
                ", dr="         + str(dr) + 
                ", epoch="      + str(epochs) + 
                ", batch_size=" + str(batch) +
                ", L:"          + str(layers) +
                ", N:"          + str(nodes))
        
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipvalue=1)

        # Build a model
        oModel = OpioidModel(X[0].shape, layers=layers, nodes=nodes, dropout_rate=dr)

        # Optimizer values
        oModel.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', mean_pred])

        ## Add a callback for saving weights each epoch
        filepath = "Weights/Weights+lr" + str(lr) + "+dr" + str(dr) + "+bz" + str(batch) + "+n" + str(nodes) + "+l" + str(layers) +".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = oModel.fit(X_trainDev, Y_trainDev, epochs=epochs, shuffle=False, batch_size=batch, callbacks=callbacks_list, validation_split=validation_split)

        #oModel.save(weightfile)

        # Main results
        train_predictions = oModel.evaluate(X_train, Y_train, verbose=0)
        dev_preds = oModel.evaluate(X_dev, Y_dev, verbose=0)
        test_preds = oModel.evaluate(X_test, Y_test, verbose=0)

        # Save relevant data to csv files
        csvFileName = "Results/history+lr" + str(lr) + "+dr" + str(dr) + "+bz" + str(batch)  + "+n" + str(nodes) + "+l" + str(layers)+ ".csv"
        mergedData = np.column_stack((
                np.array(history.history["loss"]),
                np.array(history.history["val_loss"]),
                np.array(history.history['binary_accuracy']),
                np.array(history.history['val_binary_accuracy'])))

        np.savetxt(csvFileName, mergedData, delimiter=",", header="loss, val_loss, binary_accuracy, val_binary_accuracy", comments="")

#else:
#    oModel = load_model(weightfile, custom_objects={'mean_pred': mean_pred})

########################################################################################################################
# This section need to be a file scanner)
#
########################################################################################################################
#resultsSumm = np.zeros((len(learning_rates)*len(dropout_rates)*len(batch_sizes)*len(node_sizes)*len(layer_sizes), 11))

#row = 0
#for i, lr in enumerate(learning_rates):
#    for j, dr in enumerate(dropout_rates):
#        for k, batch_sz in enumerate(batch_sizes):
#            for l, nodes in enumerate(node_sizes):
#                for m, layers in enumerate(layer_sizes):

#                    resultsSumm[row, 0] = lr
#                    resultsSumm[row, 1] = dr
#                    resultsSumm[row, 2] = batch_sz
#                    resultsSumm[row, 3] = nodes
#                    resultsSumm[row, 4] = layers

#                    resultsSumm[row, 5] = trainingLoss[i, j, k, l, m]
#                    resultsSumm[row, 6] = trainingAccuracy[i, j, k, l, m]
#                    resultsSumm[row, 7] = devLoss[i, j, k, l, m]
#                    resultsSumm[row, 8] = devAccuracy[i, j, k, l, m]
#                    resultsSumm[row, 9] = testLoss[i, j, k, l, m]
#                    resultsSumm[row, 10] = testAccuracy[i, j, k, l ,m]
#                    row += 1


#csvFileName = "Results/summary.csv"

#format = []
#format.append ("%7.7f")
#format.append ("%3.1f")
#format.append ("%5d")
#format.append ("%5d")
#format.append ("%5d")

#format.append ("%6.3f")
#format.append ("%5.3f")
#format.append ("%6.3f")
#format.append ("%5.3f")
#format.append ("%6.3f")
#format.append ("%5.3f")


#np.savetxt(csvFileName, resultsSumm, delimiter=",",
#           header="learning_rate, dropout_rate, batch_size, nodes, layers, trainingLoss, trainingAccuracy, devLoss, devAccuracy, testLoss, testAccuracy",
#           comments="",
#           fmt=format
#           )



