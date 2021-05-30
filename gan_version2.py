import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import tensorflow.keras.backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory as idfd
from PIL import Image
from GAN_model import *
from prepare_dataset import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

count = 0    
EPOCHS=100
LAMBDA_whole = 0.3
LAMBDA_mask = 0.7
LAMBDArc = 100

gen = generatore()
#gen.summary()
disc_whole = disc_whole_region()
disc_whole_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_whole.compile(disc_whole_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
disc_whole.summary()

disc_mask = disc_mask_region()
disc_mask_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_mask.compile(disc_mask_optimizer, loss="binary_crossentropy", metrics=["accuracy"])
disc_mask.summary()

INIT_LR=2e-4
input_mask = Input((128, 128, 3))
input_map = Input((128, 128, 1))
discOutput = disc_whole(gen([input_mask, input_map]))
gan = Model(inputs=[input_mask, input_map], outputs=discOutput)
ganOpt = tf.keras.optimizers.Adam(lr=INIT_LR, beta_1=0.5)
gan.compile(loss="binary_crossentropy", optimizer=ganOpt)
gan.summary()

# vgg_model = vgg19_model()
print("Modelli caricati")

train_ds, test_ds = prepare_tf_GAN()
print("Dati caricati")
#---------LOSS PERCEPTUAL-----------

# @tf.function
# def perceptual_loss(gen_image, gt_image):
#     h1_list = vgg_model(gen_image)
#     h2_list = vgg_model(gt_image)

#     perc_loss = 0.0
#     for h1, h2 in zip(h1_list, h2_list):
#         h1 = K.batch_flatten(h1)
#         h2 = K.batch_flatten(h2)
#         perc_loss += K.sum(K.square(h1 - h2), axis=-1)
    
#     return perc_loss

#---------LOSS GAN------------------

print("Ottimizzatori caricati")

history = {}
#keep track of loss and accuracy separately for true and fake images
history['G_loss'] = []
history['D_loss_true'] = []
history['D_loss_fake'] = []
accuracy = {}
accuracy['Acc_true'] = []
accuracy['Acc_fake'] = []

def fit(train_ds, epochs, test_ds):
    for epoch in range(EPOCHS):
        for example_input, example_map, example_target in test_ds.take(1):
            generate_images(gen, example_input, example_map, example_target)
        print("Epoch: ", epoch)   
        for n, (input_image, input_map, target) in train_ds.enumerate():
           
            batch_size = input_image.shape[0]
            # print(n)
            y = tf.ones([batch_size, 14, 14, 1])
            # print(y.shape)
            discLoss, discAcc = disc_whole.train_on_batch(x=target, y=y)
            history['D_loss_true'].append(discLoss)          
            accuracy['Acc_true'].append(discAcc)
            
            # generate some fake samples
            genImages=gen.predict([input_image, input_map])
            y = tf.zeros([batch_size, 14, 14, 1])
            
            discLoss, discAcc = disc_whole.train_on_batch(x=genImages, y=y)
            history['D_loss_fake'].append(discLoss)          
            accuracy['Acc_fake'].append(discAcc)

            # some authors suggest randomly flipping some labels to introduce random variations
            fake_labels = tf.ones([batch_size, 14, 14, 1])
            ganLoss = gan.train_on_batch(x=[input_image, input_map], y=fake_labels)
            history['G_loss'].append(ganLoss)
    
        # at the end of each epoc  
        print("epoch " + str(epoch) + ": discriminator loss " + str(discLoss)+  " ( "  + str(discAcc) + " ) - generator loss " + str(ganLoss))

def generate_images(model, test_input, test_map, tar):
    prediction = model([test_input, test_map], training=True)
    fig=plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Test Input', 'Ground Truth', 'Predicted Image']
    global count
    count += 1
    plt.suptitle("Epoch" + str(count))
    stringa ="Risultati\\GAN\\20k_100epoch\\" + str(count) + ".png"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    fig.savefig(stringa)

fit(train_ds, EPOCHS, test_ds)