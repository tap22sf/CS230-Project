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

# General
seed = 0
training_portion = 0.9
test_portion = 0.1
validation_split = 1/9

# Read from folder "Medical Database" which is at the same level as the project folder
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(fileDir)))
dataFile1Path = os.path.join(parentDir, 'New Medical Database/scaled_PCs')
dataFile2Path = os.path.join(parentDir, 'New Medical Database/data_y')

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
X_dev = X_trainDev[int((1 - validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]
Y_dev = Y_trainDev[int((1 - validation_split) * X_trainDev.shape[0]): X_trainDev.shape[0], :]

print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_dev shape: " + str(X_dev.shape))
print("Y_dev shape: " + str(Y_dev.shape))

csvFileName = "Results/trainingData_X.csv"
np.savetxt(csvFileName, X_train, delimiter=",")

csvFileName = "Results/trainingData_Y.csv"
np.savetxt(csvFileName, Y_train, delimiter=",")

csvFileName = "Results/devData_X.csv"
np.savetxt(csvFileName, X_dev, delimiter=",")

csvFileName = "Results/devData_Y.csv"
np.savetxt(csvFileName, Y_dev, delimiter=",")

csvFileName = "Results/testData_X.csv"
np.savetxt(csvFileName, X_test, delimiter=",")

csvFileName = "Results/testData_Y.csv"
np.savetxt(csvFileName, Y_test, delimiter=",")