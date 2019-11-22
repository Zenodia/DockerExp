from __future__ import print_function
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from tensorflow.keras.layers import BatchNormalization , Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
# Create a new model instance

smooth = 1.

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

### customized unet loss function in order to ensure sucess and fast training 
def combined_dice_binary_loss(y_true,y_pred):
    def dice_loss(y_true,y_pred):
        numerator= 2 * tf.reduce_sum( y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true +y_pred, axis=(1,2,3))
        return tf.reshape(1-numerator/denominator, (-1,1,1))
    return binary_crossentropy(y_true,y_pred)+dice_loss(y_true,y_pred)
def get_unet():
    inputs = Input((96, 96, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

data_path = os.environ['data_path']
EPOCHS =int(os.environ['epoch'])
batch_size=int(os.environ['batch_size'])
checkpoint_path=os.environ['model_save']
load_model=os.environ['load_model']

print("data_path ", data_path)
print("EPOCHES ", EPOCHS)
print("batch_size ", batch_size)
print("load_model ", load_model)
print("model saving ", checkpoint_path)
train=np.load(data_path+'/images.npy')
train_mask=np.load(data_path+'/masks.npy')
train = train.astype('float32')
mini = np.min(train)  # mean for data centering
maxi = np.max(train)  # std for data normalization
train -= mini
train /= (maxi -mini)
train_mask = train_mask.astype('float32')
mask_max=np.max(train_mask)
train_mask/=mask_max
# scale masks to [0, 1]
#print(train.shape, train_mask.shape)
print("train min",np.min(train), "train max",np.max(train), "\n mask min",np.min(train_mask), "\n mask max", np.max(train_mask))

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=99)

model=get_unet()

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor='val_loss',
    save_best_only=True)

model_history = model.fit(train, train_mask, batch_size=batch_size, epochs=EPOCHS, verbose=1, shuffle=True,validation_data=(train,train_mask),
          callbacks=[cp_callback,early_stopping_callback])

# Save the weights
#model.save_weights('./ngc_checkpoint_weight')
