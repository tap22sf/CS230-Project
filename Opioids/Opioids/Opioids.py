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
training_portion = 0.1
test_portion = 0.01
validation_split = 0.1


weightDir = 'Weights'

resultsDir = 'Results'
seed = 0 # Random generator

# Optimization parameters

#epochs=1
epochs = 2
batch_sizes = [10000, 20000]
loss = "binary_crossentropy"
#loss = "mean_squared_error"

# Regularization
# dropout_rates = [0, 0.1, 0.2, 0.3]
dropout_rates = [0, 0.1]

# Adam parameters
# r = [0, -1, -2, -3, -4]
r = [0, -1]
learning_rates = np.power(float(10), np.array(r))

lr = 0.002
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)


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
X, Y = loadData(dataFile1Path, dataFile2Path)
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
X_dev = X_trainDev[(1 - int(validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]
Y_dev = Y_trainDev[(1 - int(validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]

np.savetxt (resultsDir + r'\Test_y.csv', Y_test)

# Initialization of variables

trainingLoss = np.zeros((len(learning_rates), len(dropout_rates), len(batch_sizes)))
trainingAccuracy = np.zeros(trainingLoss.shape)
devLoss = np.zeros(trainingLoss.shape)
devAccuracy = np.zeros(trainingLoss.shape)
testLoss = np.zeros(trainingLoss.shape)
testAccuracy = np.zeros(trainingLoss.shape)

np_loss_history = np.zeros((len(learning_rates), len(dropout_rates), len(batch_sizes), epochs))
np_val_loss_history = np.zeros(np_loss_history.shape)
np_binary_accuracy_history = np.zeros(np_loss_history.shape)
np_val_binary_accuracy_history = np.zeros(np_loss_history.shape)

# Train the model, iterating on the data in batches of 32 samples
if train:
    for i, lr in enumerate(learning_rates):
        for j, dr in enumerate(dropout_rates):
            for k, batch_sz in enumerate(batch_sizes):
                print()
                print("Running model with lr=" + str(lr) + ", dr=" + str(dr) + ", batch_size=" + str(batch_sz))

                optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipvalue=1)
                weightfile = "Weights/omodel_weights-lr" + str(lr) + "-dr" + str(dr) + ".h5"

                # Build a model
                oModel = OpioidModel(X[0].shape, dr)

                # Optimizer values
                oModel.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', mean_pred])

                ## Add a callback for saving weights each epoch
                # checkpoint
                filepath = "Weights/Weights-lr" + str(lr) + "-dr" + str(dr) + "-batchSize" + str(batch_sz) + ".hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
                callbacks_list = [checkpoint]

                history = oModel.fit(X_trainDev, Y_trainDev, epochs=epochs, shuffle=False, batch_size=batch_sz, callbacks=callbacks_list, validation_split=validation_split)

                oModel.save(weightfile)

                # Main results
                preds = oModel.evaluate(X_train, Y_train, verbose=0)
                trainingLoss[i, j, k] = preds[0]
                trainingAccuracy[i, j, k] = preds[1]
                # Y_pred_train = oModel.predict(X_train, batch_size=None, verbose=0, steps=None)

                preds = oModel.evaluate(X_dev, Y_dev, verbose=0)
                devLoss[i, j, k] = preds[0]
                devAccuracy[i, j, k] = preds[1]
                # Y_pred_dev = oModel.predict(X_dev, batch_size=None, verbose=0, steps=None)

                preds = oModel.evaluate(X_test, Y_test, verbose=0)
                testLoss[i, j, k] = preds[0]
                testAccuracy[i, j, k] = preds[1]
                # Y_pred_test = oModel.pred,ict(X_test, batch_size=None, verbose=0, steps=None)


                # Save relevant data to csv files
                np_loss_history = np.array(history.history["loss"])
                np_val_loss_history = np.array(history.history["val_loss"])
                np_binary_accuracy_history = np.array(history.history['binary_accuracy'])
                np_val_binary_accuracy_history = np.array(history.history['val_binary_accuracy'])

                csvFileName = "Results/history-lr" + str(lr) + "-dr" + str(dr) + "-batchSize" + str(batch_sz) + ".csv"
                mergedData = np.column_stack((np_loss_history, np_val_loss_history,
                                              np_binary_accuracy_history, np_val_binary_accuracy_history))
                np.savetxt(csvFileName, mergedData, delimiter=",", header="loss, val_loss, binary_accuracy, val_binary_accuracy", comments="")
else:
    oModel = load_model(weightfile, custom_objects={'mean_pred': mean_pred})

########################################################################################################################

resultsSumm = np.zeros((len(learning_rates)*len(dropout_rates)*len(batch_sizes), 9))

for i, lr in enumerate(learning_rates):
    for j, dr in enumerate(dropout_rates):
        for k, batch_sz in enumerate(batch_sizes):
            row = (i * len(dropout_rates) * len(batch_sizes)) + (j * len(batch_sizes)) + k

            resultsSumm[row, 0] = lr
            resultsSumm[row, 1] = dr
            resultsSumm[row, 2] = batch_sz
            resultsSumm[row, 3] = trainingLoss[i, j, k]
            resultsSumm[row, 4] = trainingAccuracy[i, j, k]
            resultsSumm[row, 5] = devLoss[i, j, k]
            resultsSumm[row, 6] = devAccuracy[i, j, k]
            resultsSumm[row, 7] = testLoss[i, j, k]
            resultsSumm[row, 8] = testAccuracy[i, j, k]

csvFileName = "Results/summary.csv"
np.savetxt(csvFileName, resultsSumm, delimiter=",",
           header="learning_rate, dropout_rate, batch_size, trainingLoss, trainingAccuracy, devLoss, devAccuracy, testLoss, testAccuracy",
           comments="")


# oModel.summary()
# print()
#
#
# # Evaluate the results
# if train:
#     preds = oModel.evaluate(X_trainDev, Y_trainDev)
#     print("Training Loss = " + str(preds[0]))
#     print("Training Accuracy = " + str(preds[1]))
#
#     Y_pred_train = oModel.predict(X_trainDev, batch_size=None, verbose=0, steps=None)
#     print("Y Pred train Mean : " + str(Y_pred_train.mean()))
#     np.savetxt(resultsDir + r'\Pred_train_y.csv', Y_pred_train)
#
# # Evaluate model on test set
# preds = oModel.evaluate(X_test, Y_test)
# print("Test Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))
#
# Y_pred = oModel.predict(X_test, batch_size=None, verbose=0, steps=None)
# Y_pred_binary = (Y_pred > 0.5)
#
# print("Y Predictions")
# print("Y Pred Mean : " + str(Y_pred.mean()))
# np.savetxt(resultsDir + r'\Pred_y.csv', Y_pred)
#
# print("Y Predictions - binary")
# print("Y Pred Binary Mean : " + str(Y_pred_binary.mean()))
# np.savetxt(resultsDir + r'\Pred_binary_y.csv', Y_pred_binary)

# summarize history for accuracy
# if train:
#     print(history.history.keys())
#
#     plt.plot(history.history['binary_accuracy'])
#     plt.plot(history.history['val_binary_accuracy'])
#
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     plt.savefig(resultsDir + r'\Accuracy.png')
#
#     plt.title('model loss')
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     plt.savefig(resultsDir + r'\Loss.png')


