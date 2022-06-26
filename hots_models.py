#Autoencoder models to reduce dimensionality of hero stats

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers


#Single step encoder (i.e. 1 hidden layer)
def auto1(input_dim,encoding_dim):
    print('Input dim: ',input_dim)
    print('Encoding dim: ',encoding_dim)

    Xin = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(Xin)
    decoded = layers.Dense(input_dim, activation='relu')(encoded)

    autoencoder = keras.Model(Xin, decoded)

    encoder = keras.Model(Xin,encoded)

    autoencoder.compile(optimizer='adam',loss='mean_squared_error')
    
    return autoencoder, encoder


#Two step encoder; 3 hidden layers
def auto2(input_dim,encoding_dim1, encoding_dim2):
    print('Input dim: ',input_dim)
    print('Encoding dim: ',encoding_dim1,encoding_dim2)

    Xin = keras.Input(shape=(input_dim,))
    encoded1 = layers.Dense(encoding_dim1,activation='relu')(Xin)
    encoded2 = layers.Dense(encoding_dim2, activation='relu')(encoded1)
    decoded1 = layers.Dense(encoding_dim1, activation='relu')(encoded2)
    decoded2 = layers.Dense(input_dim, activation='relu')(decoded1)

    autoencoder = keras.Model(Xin, decoded2)

    encoder = keras.Model(Xin,encoded2)

    autoencoder.compile(optimizer='adam',loss='mean_squared_error')
    
    return autoencoder, encoder


