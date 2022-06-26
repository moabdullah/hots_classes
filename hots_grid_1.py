#Grid scan for auto1 model
import numpy as np
import tensorflow as tf
import pickle

import hots_models

#Load data
norm_stats = np.load('norm_stats.npz')['norm_stats']

#Remove #Games [0], Win rate [1], Game duration [2] (superfluous) and T/D [3] (redundant)
#To exclude internal stats: [-3] DPS, [-2] Range, [-1] HP
exclusions = [0,1,2,3]
X = np.delete(norm_stats,exclusions,1)
input_dim = X.shape[1]

#Try each model multiple times
trials = 20
epochs = 1000
grid = np.array([2,3,4,5,6])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=20,
                                            restore_best_weights=True)

output = [exclusions]

for i in grid:
    #Save the output and loss of each trial
    loss_list = np.zeros(trials)
    Xenc_list = np.zeros([trials,90,i])
    for n in range(trials):
        print('Trial = ',n)
        model, encoder = hots_models.auto1(input_dim=input_dim,
                                 encoding_dim = i)

        hist = model.fit(X,X, epochs = epochs ,batch_size = 90,
                        callbacks = [callback])
        loss_list[n] = hist.history['loss'][-1]
        Xenc_list[n] = encoder.predict(X)
    #Pick out the smallest loss
    argmin = np.argmin(loss_list)
    loss  = loss_list[argmin]
    Xenc = Xenc_list[argmin]
    
    output.append([i,loss,Xenc])

pickle.dump(output,open('auto1_exp0.pickle','wb'))