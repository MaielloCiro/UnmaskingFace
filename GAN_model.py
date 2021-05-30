import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

def se_block(in_block, ch, ratio=16):
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch//ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
    return Multiply()([in_block, x])

def generatore():
    input_size = (128, 128, 3)
    
    input_mask = Input(input_size)
    input_map = Input((128, 128, 1))
    inputs = concatenate([input_mask, input_map])

    conv1 = Conv2D(64, 3, activation = 'relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, 3, padding='same')(pool1)#pool2
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, padding='same')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    # se = se_block(pool1, ch=1) #pool2 <-------------------ATTENZIONE A QUESTO QUI

    # conv3 = Conv2D(256, 3, padding='same')(pool2)
    # conv3 = LeakyReLU()(conv3)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Conv2D(256, 3, padding='same')(conv3)
    # conv3 = LeakyReLU()(conv3)
    # conv3 = BatchNormalization()(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)

    # conv4 = Conv2D(512, 3, padding='same')(pool3)
    # conv4 = LeakyReLU()(conv4)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = Conv2D(512, 3, padding='same')(conv4)
    # conv4 = LeakyReLU()(conv4)
    # conv4 = BatchNormalization()(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(256, 3, padding='same', dilation_rate=(2, 2))(pool2)#pool4
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, padding='same', dilation_rate=(4, 4))(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, padding='same', dilation_rate=(8, 8))(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, padding='same', dilation_rate=(16, 16))(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = BatchNormalization()(conv5)

    # up6 = Conv2DTranspose(512, 3, strides=(2, 2), padding='same')(conv5)
    # merge6 = concatenate([conv4, up6])
    # conv6 = Conv2D(512, 3, padding='same')(merge6)
    # conv6 = LeakyReLU()(conv6)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = Conv2D(512, 3, padding='same')(conv6)
    # conv6 = LeakyReLU()(conv6)
    # conv6 = BatchNormalization()(conv6)

    # up7 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(conv6)#conv6
    # merge7 = concatenate([conv3, up7])
    # conv7 = Conv2D(256, 3, padding='same')(merge7)
    # conv7 = LeakyReLU()(conv7)
    # conv7 = BatchNormalization()(conv7)
    # conv7 = Conv2D(256, 3, padding='same')(conv7)
    # conv7 = LeakyReLU()(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv5)#conv7
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, padding='same')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, padding='same')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv8)#conv8
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='tanh', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='tanh', padding='same')(conv9)
    conv9 = Conv2D(3, 3, activation='tanh', padding='same')(conv9)
    conv9 = Conv2D(3, 1, activation='tanh')(conv9)

    generator = Model(inputs=[input_mask, input_map], outputs=conv9)
    #generator.summary()
    #plot_model(model, show_shapes=True, to_file='unet_model.png')
    #generator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return generator

def disc_whole_region():
    initializer = tf.random_normal_initializer(0., 0.02)
    input_size = (128, 128, 3)

    input = Input(input_size)
    conv1 = Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(input)
    leaky_relu1 = LeakyReLU()(conv1)

    conv2 = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(leaky_relu1)
    batchnorm2 = BatchNormalization()(conv2)
    leaky_relu2 = LeakyReLU()(batchnorm2)

    conv3 = Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(leaky_relu2)
    batchnorm3 = BatchNormalization()(conv3)
    leaky_relu3 = LeakyReLU()(batchnorm3)

    zero_pad4 = ZeroPadding2D()(leaky_relu3)  #18*18
    conv4 = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad4)  #15*15
    batchnorm4 = BatchNormalization()(conv4)
    leaky_relu4 = LeakyReLU()(batchnorm4)

    zero_pad5 = ZeroPadding2D()(leaky_relu4)  # 17*17
    conv5 = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad5)  #14*14

    discriminator = Model(inputs=input, outputs=conv5)
    return discriminator

def disc_mask_region():
    initializer = tf.random_normal_initializer(0., 0.02)
    input_size = (128, 128, 3)

    Igt_Iedit = Input(input_size)
    Imask_map = Input((128, 128, 1))
    Iinput = Input(input_size)
    input = Lambda(prepare_input_disc_mask)([Igt_Iedit, Imask_map, Iinput])

    conv1 = Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(input)
    leaky_relu1 = LeakyReLU()(conv1)

    conv2 = Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(leaky_relu1)
    batchnorm2 = BatchNormalization()(conv2)
    leaky_relu2 = LeakyReLU()(batchnorm2)

    conv3 = Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(leaky_relu2)
    batchnorm3 = BatchNormalization()(conv3)
    leaky_relu3 = LeakyReLU()(batchnorm3)

    zero_pad4 = ZeroPadding2D()(leaky_relu3)  #18*18
    conv4 = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad4)  #15*15
    batchnorm4 = BatchNormalization()(conv4)
    leaky_relu4 = LeakyReLU()(batchnorm4)

    zero_pad5 = ZeroPadding2D()(leaky_relu4)  # 17*17
    conv5 = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad5)  #14*14

    discriminator = Model(inputs=[Igt_Iedit, Imask_map, Iinput], outputs=conv5)

    #discriminator.summary()
    #plot_model(discriminator, show_shapes=True)
    return discriminator

def prepare_input_disc_mask(x):
    Igt_Iedit = x[0]
    Imask_map = x[1]
    Iinput = x[2]
    Imask_map = Imask_map/255.0
    complementary = 1-Imask_map
    firstmul = Multiply()([Iinput, complementary])
    secondmul = Multiply()([Igt_Iedit, Imask_map])
    Imask_region = Add()([firstmul, secondmul])
    return Imask_region

def vgg19_model():
    selected_layers = ["block3_conv4", "block4_conv4", "block5_conv4"]
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg.trainable = False
    outputs = [vgg.get_layer(l).output for l in selected_layers]
    vgg_model = Model(vgg.input, outputs)
    #vgg_model.summary()
    return vgg_model