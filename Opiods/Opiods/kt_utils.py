# 
# Utilities for data file loading and conversion
#

import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os.path


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
#
# Reads an input dataset and converts to np binary for faster futuer loads
#

def load_dataset (
    xfilename = r'..\..\Example Medical Database\df_x_withGeo',
    yfilename = r'..\..\Example Medical Database\data_y'):

    # Check for the real Medical Records Database in binary form
    xCsvFilename = xfilename + ".csv"
    XNpFilename = xfilename + ".npy"

    # Look for the binary version first
    if not os.path.isfile(XNpFilename):
        
        # Load the csv file
        print ("Loading input file : " + xCsvFilename)
        try :
            data_x = np.genfromtxt(xCsvFilename, skip_header=0, names=True, delimiter=',', max_rows=10000)
            np.save(XNpFilename, data_x);
            print ("Saved binary version of input file : " + xCsvFilename)
        except:
            print ("Unable to read : " + xCsvFilename)


    else:
        print ("Loading binary input file : " + XNpFilename)
        try:
            data_x = np.load(XNpFilename)
        except:
            print ("Unable to open : " + xCsvFilename)

    return (data_x)

    #train_dataset = h5py.File('datasets/train_happy.h5', "r")
    #train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    #train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    #test_dataset = h5py.File('datasets/test_happy.h5', "r")
    #test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    #test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    #classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    #train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    #test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    #return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

