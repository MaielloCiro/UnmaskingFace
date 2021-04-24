import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory as idfd
from PIL import Image
from GAN_model import *

gen = generatore()
#gen.summary()
disc_whole = disc_whole_region()
#disc_whole.summary()
disc_mask = disc_mask_region()
#disc_mask.summary()

selected_layers = ["block3_conv4", "block4_conv4", "block5_conv4"]
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False
outputs = [vgg.get_layer(l).output for l in selected_layers]
vgg_model = Model(vgg.input, outputs)
#vgg_model.summary()

#---------LOSS PERCEPTUAL-----------

@tf.function
def perceptual_loss(gen_image, gt_image):
    h1_list = vgg_model(gen_image)
    h2_list = vgg_model(gt_image)

    perc_loss = 0.0
    for h1, h2 in zip(h1_list, h2_list):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        perc_loss += K.sum(K.square(h1 - h2), axis=-1)
    
    return perc_loss

#---------LOSS GAN-----------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#gan = disc_whole(gen(input)) per 2/5

#gan1 = disc_whole(gen(input)) per 3/5
#gan2 = disc_mask(gen(input)) per 3/5

def disc_loss(disc_real_output, disc_gen_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output) #parte reale
    fake_loss = cross_entropy(tf.zeros_like(disc_gen_output), disc_gen_output) #parte generata
    total_loss = -(real_loss + fake_loss)
    return total_loss

def gen_loss(disc_gen_output):
    adv_loss = -cross_entropy(tf.ones_like(disc_gen_output), disc_gen_output) #loss adversarial
    return adv_loss

def rec_loss(gen_output, Igt):
    l1_loss = tf.reduce_mean(tf.abs(Igt - gen_output)) #loss L1
    SSIM_loss = 1 - tf.image.ssim(gen_output, Igt) #loss SSIM
    rc_loss = l1_loss + SSIM_loss
    return rc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_whole_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_mask_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
LAMBDA_whole = 0.3
LAMBDA_mask = 0.7
LAMBDArc = 100

@tf.function
def first_train_step(input_image, Igt, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = gen(input_image, training=True)

        real_output = disc_whole(input_image, training=True)
        fake_output = disc_whole(generated_image, training=True)

        gen_loss = gen_loss(fake_output)
        disc_loss = disc_loss(real_output, fake_output)
        rc_loss = rec_loss(generated_image, Igt)
        perc_loss = perceptual_loss(generated_image, Igt)
        gen_tot_loss = LAMBDArc(rc_loss + perc_loss) + gen_loss

    gradients_of_generator = gen_tape.gradient(gen_tot_loss, gen.trainable_variables)
    gradients_of_disc_whole = disc_tape.gradient(disc_loss, disc_whole.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_whole_optimizer.apply_gradients(zip(gradients_of_disc_whole, disc_whole.trainable_variables))

def second_train_step(input_image, Igt, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = gen(input_image, training=True)

        real_output_whole = disc_whole(input_image, training=True)
        fake_output_whole = disc_whole(generated_image, training=True)
        real_output_mask = disc_mask(input_image, training=True)
        fake_output_mask = disc_mask(generated_image, training=True)

        gen_loss_whole = gen_loss(fake_output_whole)
        disc_loss_whole = disc_loss(real_output_whole, fake_output_whole)
        gen_loss_mask = gen_loss(fake_output_mask)
        disc_loss_mask = disc_loss(real_output_mask, fake_output_mask)
        rc_loss = rec_loss(generated_image, Igt)
        perc_loss = perceptual_loss(generated_image, Igt)
        gen_tot_loss = LAMBDArc(rc_loss + perc_loss) + LAMBDAwhole(gen_loss_whole) + LAMBDAmask(gen_loss_mask)

    gradients_of_generator = gen_tape.gradient(gen_tot_loss, gen.trainable_variables)
    gradients_of_disc_whole = disc_tape.gradient(disc_loss_whole, disc_whole.trainable_variables)
    gradients_of_disc_mask = disc_tape.gradient(disc_loss_mask, disc_mask.trainable_variables)


    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_whole_optimizer.apply_gradients(zip(gradients_of_disc_whole, disc_whole.trainable_variables))
    disc_mask_optimizer.apply_gradients(zip(gradients_of_disc_mask, disc_mask.trainable_variables))

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):

    # for example_input, example_target in test_ds.take(1):
    #   generate_images(generator, example_input, example_target)
    # print("Epoch: ", epoch)

    # Train
        for n, (input_image, target) in train_ds.enumerate():
            if epoch < first_epochs:
                first_train_step(input_image, target, epoch)
            else:
                second_train_step(input_image, target, epoch)

    # saving (checkpoint) the model every 20 epochs
        # if (epoch + 1) % 20 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        #     print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
        #     checkpoint.save(file_prefix=checkpoint_prefix)