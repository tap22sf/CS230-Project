# 
# Utilities for data file loading and conversion
#

import math
import numpy as np

from sklearn import preprocessing

import pandas as pd

from dataFields import *



def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
#
# Reads an input dataset and converts to np binary for faster futuer loads
#

def weekdayToInt(weekday):
    return {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    } [weekday]


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

    return (loadDataSet (xFilename, xusecols, norm=True), loadDataSet (yFilename, yusecols, yconverters))

# Check for the real Medical Records Database in binary form
def loadDataSet (filename, usecols, norm=False):
    CSVFilename = filename + ".csv"
    NPFilename = filename + ".npy"
    
    with open(CSVFilename, 'r') as f:
        print ("Loading input file : " + CSVFilename)
        dfr = pd.read_csv(CSVFilename, usecols=usecols)
        
        df = dfr.apply(pd.to_numeric, errors='coerce')
        dfz = df.fillna(0)

        data = dfz.values
        
        if norm:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(data)

            df_norm = pd.DataFrame(x_scaled)
            print (df_norm)
            return df_norm.values
        else:
            print (dfz)
            return dfz.values


