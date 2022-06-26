#Grid scan for the auto2 model
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

trials = 20
epochs = 1000
grid = np.array([5])
dim2 = 8  #largest dim of 1st encoder


callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=20,
                                            restore_best_weights=True)

output = [exclusions]

for i in grid:
    for j in np.arange(i+1,dim2+1):
        loss_list = np.zeros(trials)
        Xenc_list = np.zeros([trials,90,i])
        for n in range(trials):
            print('Trial = ',n)
            model, encoder = hots_models.auto2(input_dim=input_dim,
                                     encoding_dim1 = j,
                                     encoding_dim2 = i)

            hist = model.fit(X,X, epochs = epochs ,batch_size = 90,
                            callbacks = [callback])
            loss_list[n] = hist.history['loss'][-1]
            Xenc_list[n] = encoder.predict(X)
        argmin = np.argmin(loss_list)
        loss  = loss_list[argmin]
        Xenc = Xenc_list[argmin]

        output.append([i,j,loss,Xenc])

pickle.dump(output,open('auto2_exp0.pickle','wb'))