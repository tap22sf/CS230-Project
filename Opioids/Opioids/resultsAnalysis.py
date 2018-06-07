#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
#
import os
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


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


###################################################################################################
# Parameters

# General
train = True
training_portion = .30
test_portion = 0.01

# Amount of training set to holdout for validation (dev set)
validation_split = 0.1

weightDir = 'Weights'
resultsDir = 'Results'

seed = 0  # Random generator

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


# Hyperparameters - list of dictionaries to setup training runs
parameters = []

# Node size and layer sensitivity testing
# for n in (500, 1000, 2000, 3000):
#    for l in (0, 1, 2):
#        lr = -3.0
#        epochs = 100
#        dropout = 0.5
#        run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':l,'nodes':n, 'dropout':dropout}
#        parameters.append (run)


## Learning rate senstivity tests
# for lr in (-2, -3, -4, -5):
#    layers = 2
#    nodes = 2000
#    epochs = 100
#    dropout = 0.5
#    run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':layers,'nodes':nodes, 'dropout':dropout}
#    parameters.append (run)

# Learning rate sensitivity tests
lr = -3
layers = 2
nodes = 2000
epochs = 50
epochs = 2
dropout = 0.5

# batch size scan
# for bz in (32, 64, 128, 256, 512, 1024, 2048, 4096):
for bz in (2048, 4096):
    run = {'epochs': epochs, 'batch': bz, 'lr': lr, 'layers': layers, 'nodes': nodes, 'dropout': dropout}
    parameters.append(run)

##########################################################################################&#########

resultsSumm = np.zeros((len(parameters), 11))
row = 0

# Train the model, iterating on the data in batches based on the run database
if train:
    for run in parameters:
        # Extract and adjust run variables
        epochs = run['epochs']
        batch = run['batch']
        lr = run['lr']
        lr = np.power(float(10), np.array(lr))
        layers = run['layers']
        nodes = run['nodes']
        dr = run['dropout']

        print("=== New run" +
              "  lr=" + str(lr) +
              ", dr=" + str(dr) +
              ", epoch=" + str(epochs) +
              ", batch_size=" + str(batch) +
              ", L:" + str(layers) +
              ", N:" + str(nodes))

        # Save relevant data to csv files
        basefilename = "lr" + str(lr) + "+dr" + str(dr) + "+bz" + str(batch) + "+n" + str(nodes) + "+l" + str(layers)

        # Evaluate metrics
        print("Metric Calculations")
        # ['loss', 'binary_accuracy', 'mean_pred']

        metric_file = resultsDir + "/metrics+" + basefilename + ".csv"

        with open(metric_file, 'r') as f:
            print("Loading input file : " + metric_file)
            dfr = pd.read_csv(metric_file)
            df = dfr.apply(pd.to_numeric, errors='coerce')
            data2 = df.values

        print("Data 2 shape: " + str(data2.shape))

        resultsSumm[row, 0] = lr
        resultsSumm[row, 1] = dr
        resultsSumm[row, 2] = batch
        resultsSumm[row, 3] = nodes
        resultsSumm[row, 4] = layers

        resultsSumm[row, 5] = data2[0, 0]
        resultsSumm[row, 6] = data2[1, 0]
        resultsSumm[row, 7] = data2[0, 1]
        resultsSumm[row, 8] = data2[1, 1]
        resultsSumm[row, 9] = data2[0, 2]
        resultsSumm[row, 10] = data2[1, 2]
        row += 1

        csvFileName = "Results/summary.csv"

        format = []
        format.append("%7.7f")
        format.append("%3.1f")
        format.append("%5d")
        format.append("%5d")
        format.append("%5d")

        format.append("%6.3f")
        format.append("%5.3f")
        format.append("%6.3f")
        format.append("%5.3f")
        format.append("%6.3f")
        format.append("%5.3f")

        np.savetxt(csvFileName, resultsSumm, delimiter=",",
                   header="learning_rate, dropout_rate, batch_size, nodes, layers, trainingLoss, trainingAccuracy, devLoss, devAccuracy, testLoss, testAccuracy",
                   comments="",
                   fmt=format
                   )






