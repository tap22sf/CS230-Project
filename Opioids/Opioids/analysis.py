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

modelName =

