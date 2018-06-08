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

from tensorflow.python import debug as tf_debug

from dataParser import *
from oModel import *

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

###################################################################################################
# Parameters

training_portion = .95
test_portion = 0.05

# Amount of training set to holdout for validation (dev set)
validation_split = 0.055

weightDir = 'Weights'
resultsDir = 'Results'

seed = 0 # Random generator

# Prameters
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)
loss = "binary_crossentropy"
#loss = "mean_squared_error"
#loss = "hinge"

# Hyperparameters - list of dictionaries to setup training runs
parameters = []

# Baseline best Architecture (overfits trainign set)
n = 500
l = 3
beta_1 = 0.90
beta_2 = 0.999
lr = -3.0
epochs = 100
dropout = 0.0
l2reg = 0.0
bz = 8192


# Architecture evalaluation
for n in (100, 200, 300, 400, 500, 600):
    for l in (1, 2, 3):
        epochs = 100
        run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':l,'nodes':n, 'dropout':dropout, 'l2reg':l2reg}
        parameters.append (run)


# Best arch


# Dropout evalaluation
#for d in range(0,6):
#    dropout = dataFile1Path/10
#    run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':l,'nodes':n, 'dropout':dropout}
#    parameters.append (run)



## Learning rate senstivity tests
#for lr in (-2, -3, -4, -5):
#    layers = 3
#    nodes = 250
#    epochs = 25
#    dropout = 0.5
#    bz = 8192
#    run = {'epochs':epochs,'batch':bz,'lr' :lr,'layers':layers,'nodes':nodes, 'dropout':dropout}
#    parameters.append (run)

# Batch size sensitivity
# Smaller dataset for batchsize testing due to test time.
#training_portion = .50
#test_portion = 0.05
#validation_split = 0.055

#for bz in (128, 256, 512, 1024, 2048, 4096, 8192):
#    lr = -3
#    layers = 3
#    nodes = 250
#    epochs = 25
#    dropout = 0.5
#    run = {'epochs':epochs,'batch':bz,'lr':lr,'layers':layers,'nodes':nodes, 'dropout':dropout}
#    parameters.append (run)



# Read from folder "Medical Database" which is at the same level as the project folder
absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(fileDir)))
dataFile1Path = os.path.join(parentDir, 'Medical Database\\scaled_PCs')
#dataFile1Path = os.path.join(parentDir, 'Medical Database\\withEncodings')
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

##########################################################################################&#########
#
# Train the model, iterating on the data in batches based on the run database
#
##########################################################################################&#########
for run in parameters:

    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)
        
    # Extract and adjust run variables                    
    epochs = run ['epochs']
    batch = run ['batch']
    lr = run['lr']
    lr = np.power(float(10), np.array(lr))
    layers = run['layers']
    nodes = run['nodes']
    dr = run['dropout']
    l2reg = run['l2reg']

    print("=== New run" +
            "  lr="         + str(lr) +
            ", dr="         + str(dr) + 
            ", l2="         + str(l2reg) + 
            ", epoch="      + str(epochs) + 
            ", batch_size=" + str(batch) +
            ", L:"          + str(layers) +
            ", N:"          + str(nodes))
        
    # Build a model
    oModel = OpioidModel(X[0].shape, layers=layers, nodes=nodes, dropout_rate=dr, l2reg=l2reg)
    oModel.summary()

    # Optimizer values
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    #optimizer = SGD()
    
    oModel.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy', mean_pred])

    ## Add a callback for saving weights each epoch
    #filepath = "Weights/Weights+lr" + str(lr) + "+dr" + str(dr) + "+bz" + str(batch) + "+n" + str(nodes) + "+l" + str(layers) +".hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
        
    # Fit the model
    #history = oModel.fit(X_trainDev, Y_trainDev, epochs=epochs, shuffle=False, batch_size=batch, callbacks=callbacks_list, validation_split=validation_split)
    history = oModel.fit(X_trainDev, Y_trainDev, epochs=epochs, shuffle=False, batch_size=batch, validation_split=validation_split)

    # Save relevant data to csv files
    basefilename = "e+" + str(epochs) + "+lr+" + str(lr) + "+dr+" + str(dr) + "+l2+" + str(l2reg) + "+bz+" + str(batch)  + "+n+" + str(nodes) + "+l+" + str(layers)
    csvFileName = resultsDir + "/history+" + basefilename + ".csv"
    mergedData = np.column_stack((
            np.array(history.history["loss"]),
            np.array(history.history["val_loss"]),
            np.array(history.history['binary_accuracy']),
            np.array(history.history['val_binary_accuracy'])))

    np.savetxt(csvFileName, mergedData, delimiter=",", header="loss, val_loss, binary_accuracy, val_binary_accuracy", comments="")
    weightfile = weightDir + "/weights+" + basefilename + ".hdf5"
    oModel.save(weightfile)
    
    doMetrics = True

    if doMetrics:
        # Evaluate metrics
        print("Metric Calculations")
        train_metrics = oModel.evaluate(X_train, Y_train, verbose=1)
        dev_metrics = oModel.evaluate(X_dev, Y_dev, verbose=1)
        test_metrics = oModel.evaluate(X_test, Y_test, verbose=1)

        mergedMetrics = np.column_stack((train_metrics, dev_metrics, test_metrics))
                
        metric_file = resultsDir + "/metrics+" + basefilename + ".csv"
        format = []
        format.append ("%5.3f")
        format.append ("%5.3f")
        format.append ("%5.3f")
        np.savetxt (metric_file, mergedMetrics, delimiter=",", header= "train, dev, test", fmt=format)

        # Save predictions
        print("Prediction Calculations")
        pred_train_name = resultsDir + "/pred_train+" + basefilename + ".csv"
        predictions_train = oModel.predict(X_train, verbose=1) 
        np.savetxt (pred_train_name, predictions_train)

        pred_dev_name = resultsDir + "/pred_dev+" + basefilename + ".csv"
        predictions_dev = oModel.predict(X_dev, verbose=1) 
        np.savetxt (pred_dev_name, predictions_dev)

        pred_test_name = resultsDir + "/pred_test+" + basefilename + ".csv"
        predictions_test = oModel.predict(X_test, verbose=1) 
        np.savetxt (pred_test_name, predictions_test)

        # known good output
        print("Writting known outputs")
        train_name = resultsDir + "/known_train+" + basefilename + ".csv"
        np.savetxt (train_name, Y_train)
        dev_name = resultsDir + "/known_dev+" + basefilename + ".csv"
        np.savetxt (dev_name, Y_dev)
        test_name = resultsDir + "/known_test+" + basefilename + ".csv"
        np.savetxt (test_name, Y_test)


