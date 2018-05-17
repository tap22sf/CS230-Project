#
# Opiods.py - Main Application to train network to detect likelihood of opiod adiction following lumbar surgey
# 

import numpy as np

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import *

import keras.backend as K

from dataParser import *
from oModel import *



# Read the datasets
X,Y = loadData(r'E:\Medical Database\scaled', r'E:\Medical Database\data_y')
#X,Y = loadData()

# Normalize input and output fields
print ("number of training examples = " + str(X.shape[0]))
print ("X shape: " + str(X.shape))
print ("X:")
print(X)

print ("number of result examples = " + str(Y.shape[0]))
print ("Y shape: " + str(Y.shape))
print ("Y:")
print(Y)
print (np.mean(Y))

# Divide into train/dev/test sets

# Build a model
oModel = OpiodModel(X[0].shape)

# Apply input
# Optimizer values
lr = 0.05
beta_1 = 0.9
beta_2 = 0.999
epsilon = 10 ** (-8)
optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)

oModel.compile(optimizer=optimizer, loss = "binary_crossentropy", metrics = ["binary_accuracy"])

# Train the model, iterating on the data in batches of 32 samples
oModel.fit(X, Y, epochs=300, batch_size=100)

# Evaluate the results
#preds = oModel.evaluate(X_test, Y_test)

#print()
#print ("Loss = " + str(preds[0]))
#print ("Test Accuracy = " + str(preds[1]))
#for layer in oModel.layers:
#    weights = layer.get_weights() # list of numpy arrays
#    print(weights)

oModel.summary()
