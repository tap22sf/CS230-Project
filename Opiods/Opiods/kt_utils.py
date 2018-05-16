# 
# Utilities for data file loading and conversion
#

import keras.backend as K
import math
import numpy as np
import os.path


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
#
# Reads an input dataset and converts to np binary for faster futuer loads
#

def loadData (
    xFilename = r'..\..\Example Medical Database\df_x_withGeo',
    yFilename = r'..\..\Example Medical Database\data_y'):
    """
    Load data from CSV files.
    
    Arguments:
    xFilename -- path to a X datafile
    yFilename -- path to a Y datafile
    
    Returns:
    xdata -- np.array(m, #inputs)
    ydata -- np.array(m, #outputs)
    """

    return (loadDataSet (xFilename), loadDataSet (yFilename))

# Check for the real Medical Records Database in binary form
def loadDataSet (filename):
    CSVFilename = filename + ".csv"
    NPFilename = filename + ".npy"
    
    if not os.path.isfile(NPFilename):
        
        # Load the csv file
        print ("Loading input file : " + CSVFilename)
        try :
            data = np.genfromtxt(CSVFilename, skip_header=0, names=True, delimiter=',', missing_values=0.0, dtype='f8', max_rows=10000)
            #np.save(NPFilename, data);
            print ("Saved binary version of input file : " + NPFilename)
        except:
            exit ("Unable to read : " + CSVFilename)
           

    else:
        print ("Loading binary input file : " + NPFilename)
        try:
            data = np.load(NPFilename)
        except:
            exit ("Unable to open : " + NPFilename)

#    return data.transpose()
    return data

