import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

x = pd.read_csv('fer2013.csv')
print(x.values.shape)

data = x.values
print(data.shape)
print(data[0, 0])

y = data[:, 0]

pixels = data[:, 1]
print(type(pixels))
print(len(pixels[0]))
print(pixels[10][10])

X = np.zeros((pixels.shape[0], 48*48))

for ix in range(X.shape[0]):
    p = pixels[ix].split(' ')
    for iy in range(X.shape[1]):
        X[ix, iy] = int(p[iy])
        
np.save('facial_data_X', X)
np.save('facial_labels', y)
