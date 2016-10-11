# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 22:25:08 2015

@author: Dan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
#import sklearn
#import statsmodels
import scipy
import keras
#from pandasql import *

from keras.models import Sequential

model = Sequential()

from keras.layers.core import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd')