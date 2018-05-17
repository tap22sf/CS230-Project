# 
# Utilities for data file loading and conversion
#

import math
import numpy as np
from sklearn import preprocessing
import pandas as pd

from dataFields import *

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

    return (loadDataSet (xFilename, xusecols, norm=False, fake_data=0), loadDataSet (yFilename, yusecols, fake_data=0))

# Check for the real Medical Records Database in binary form
def loadDataSet (filename, usecols, norm=False, fake_data=0):
    CSVFilename = filename + ".csv"
    NPFilename = filename + ".npy"
    
    with open(CSVFilename, 'r') as f:
        print ("Loading input file : " + CSVFilename)
        dfr = pd.read_csv(CSVFilename, usecols=usecols)
        
        df = dfr.apply(pd.to_numeric, errors='coerce')
        dfz = df.fillna(0)

        data = dfz.values
        
        if fake_data == 1:
            mu, sigma = 0, 2 # mean and standard deviation
            s1 = np.random.normal(mu, sigma, size=dfz.values.shape)
            
            mu, sigma = 3, 2 # mean and standard deviation
            s2= np.random.normal(mu, sigma, size=dfz.values.shape)
            data = np.concatenate((s1, s2), axis=0)

        elif fake_data == 2:
            mu, sigma = 1, 0.1 # mean and standard deviation
            s1= np.random.normal(mu, sigma, size=dfz.values.shape)
                        
            mu, sigma = 1, 0.1 # mean and standard deviation
            s2= np.random.normal(mu, sigma, size=dfz.values.shape)
            data = np.concatenate((s1, s2), axis=0)

 
        if norm:
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(data)
            df_norm = pd.DataFrame(x_scaled)
            data =  df_norm.values


        return data


