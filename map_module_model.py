import tensorflow as tf
import numpy as np
import os
#import skimage.io as io
#import skimage.transform as trans
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def modello():
    #input_size=(256, 256, 1)
    input_size = (128, 128, 1)

    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    #conv2 = Conv2D(32, 3, padding='same')(pool1)
    #conv2 = LeakyReLU()(conv2)
    #conv2 = BatchNormalization()(conv2)
    #conv2 = Conv2D(32, 3, padding='same')(conv2)
    #conv2 = LeakyReLU()(conv2)
    #conv2 = BatchNormalization()(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(64, 3, padding='same')(pool1)#pool2
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, padding='same')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(128, 3, padding='same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, padding='same')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(256, 3, padding='same')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, padding='same')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(0.5)(pool5)

    conv6 = Conv2D(512, 3, padding='same')(pool5)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, padding='same')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv5, up7])
    conv7 = Conv2D(256, 3, padding='same')(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, padding='same')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv4, up8])
    conv8 = Conv2D(128, 3, padding='same')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, padding='same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv3, up9])
    conv9 = Conv2D(64, 3, padding='same')(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, padding='same')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = BatchNormalization()(conv9)

    #up10 = Conv2DTranspose(32, 3, strides=(2, 2), padding='same')(conv9)
    #merge10 = concatenate([conv2, up10])
    #conv10 = Conv2D(32, 3, padding='same')(merge10)
    #conv10 = LeakyReLU()(conv10)
    #conv10 = BatchNormalization()(conv10)
    #conv10 = Conv2D(32, 3, padding='same')(conv10)
    #conv10 = LeakyReLU()(conv10)
    #conv10 = BatchNormalization()(conv10)

    up11 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv9)#conv10
    merge11 = concatenate([conv1, up11])
    conv11 = Conv2D(32, 3, activation='tanh', padding='same')(merge11)
    conv11 = Conv2D(32, 3, activation='tanh', padding='same')(conv11)
    conv11 = Conv2D(2, 3, activation='tanh', padding='same')(conv11)
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv11)

    model = Model(inputs=inputs, outputs=conv11)

    model.summary()
    #plot_model(model, show_shapes=True, to_file='unet_model.png')

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
