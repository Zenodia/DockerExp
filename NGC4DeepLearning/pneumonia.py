from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow using pip in conda environment , make sure to create py36 
# conda env create -n tf2_py36 python=3.6 anaconda to create the environment
# once created the environment conda activate tf2_py36 and then pip install TF2.0
# pip install tensorflow==2.0.0 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, Callback , ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt
import random

import numpy as np
X=np.load('X.npy')
X_std = X / 255.
y=np.load('Y.npy')
print(X.shape,y.shape)
from tensorflow.keras import Model
# this is the default distributed strategy API tensorflow2.0 
# provided for keras for one machine,multiple GPU training

mirrired_strategy=tf.distribute.MirroredStrategy()
with mirrired_strategy.scope():
    model = Sequential([
        Conv2D(8, 3, padding='same', activation='relu', input_shape=(150, 150 ,3)),
        #MaxPooling2D(),
        #Dropout(0.2),
        Conv2D(8, 3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Dropout(0.25),
        Conv2D(16, 3, padding='same', activation='relu'),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Dropout(0.25),
        Conv2D(24, 3, padding='same', activation='relu'),
        Conv2D(24, 3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3,3)),
        Dropout(0.25),
        Flatten(),
        Dense(300, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [
        ModelCheckpoint(
        filepath='tf2_test.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='loss',
        verbose=1),
        EarlyStopping(monitor='loss',min_delta=1e-5, patience=5)
    
]
model.fit(x=X_std, y=y, epochs=100, shuffle=True, callbacks=callbacks )
from tensorflow.keras.models import load_model
loaded_model=load_model('tf2_test.h5')
loaded_model.summary()
# use loaded model to make prediction
rn=random.randint(0,len(X)-1)
out=loaded_model.predict(X_std[rn].reshape(1,150,150,3))
true_label= 'Pneumonia' if y[rn]==1 else 'Normal'
print("true label is :", true_label )
print("predicted proba having Pneumonia", str(round(out[0][0],3)))