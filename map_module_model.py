'''
Definition Map Module
'''

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def seg_model():
    input_size = (128, 128, 1)

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, 3, padding='same')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, padding='same')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(256, 3, padding='same')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, padding='same')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(512, 3, padding='same')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, padding='same')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024, 3, padding='same')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, padding='same')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([conv4, up6])
    conv6 = Conv2D(512, 3, padding='same')(merge6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, padding='same')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, padding='same')(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, padding='same')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, padding='same')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, padding='same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='tanh', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='tanh', padding='same')(conv9)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv9)

    model.summary()
    #plot_model(model, show_shapes=True, to_file='unet_model.png')

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model
